import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class StackedRelationalGraphConvolution(torch.nn.Module):
    '''
    input:  entity features:    batch x num_entity x input_dim
            relation features:  batch x num_relations x input_dim
            adjacency matrix:   batch x num_relations x num_entity x num_entity
    '''

    def __init__(self, entity_input_dim, relation_input_dim, num_relations, hidden_dims, num_bases,
                 use_highway_connections=False, dropout_rate=0.0, real_valued_graph=False):
        super(StackedRelationalGraphConvolution, self).__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.hidden_dims = hidden_dims
        self.num_relations = num_relations
        self.dropout_rate = dropout_rate
        self.num_bases = num_bases
        self.real_valued_graph = real_valued_graph
        self.nlayers = len(self.hidden_dims)
        self.stack_gcns()
        self.use_highway_connections = use_highway_connections
        if self.use_highway_connections:
            self.stack_highway_connections()

    def stack_highway_connections(self):
        highways = [torch.nn.Linear(self.hidden_dims[i], self.hidden_dims[i]) for i in range(self.nlayers)]
        self.highways = torch.nn.ModuleList(highways)
        self.input_linear = torch.nn.Linear(self.entity_input_dim, self.hidden_dims[0])

    def stack_gcns(self):
        gcns = [RelationalGraphConvolution(self.entity_input_dim if i == 0 else self.hidden_dims[i - 1],
                                           self.relation_input_dim, self.num_relations, self.hidden_dims[i],
                                           num_bases=self.num_bases)
                for i in range(self.nlayers)]
        self.gcns = torch.nn.ModuleList(gcns)

    def forward(self, node_features, relation_features, adj):
        x = node_features
        for i in range(self.nlayers):
            if self.use_highway_connections:
                if i == 0:
                    prev = self.input_linear(x)
                else:
                    prev = x.clone()
            x = self.gcns[i](x, relation_features, adj)  # batch x num_nodes x hid
            if self.real_valued_graph:
                x = torch.sigmoid(x)
            else:
                x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)
            if self.use_highway_connections:
                gate = torch.sigmoid(self.highways[i](x))
                x = gate * x + (1 - gate) * prev
        return x


class RelationalComplEx(torch.nn.Module):
    """
    implementing the RotateE for edge prediction
    reference paper: https://arxiv.org/pdf/1606.06357
    reference code: https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding
    """

    def __init__(self, entity_input_dim, relation_input_dim, num_relations):
        super(RelationalComplEx, self).__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.num_relations = num_relations
        self.relations_weights_bilinear = Parameter(
            torch.FloatTensor(self.num_relations, (self.relation_input_dim + self.entity_input_dim)))
        self.reset_parameters()

    def forward(self, node_encodings, relation_encodings):
        re_node_encodings, im_node_encodings = torch.chunk(node_encodings, 2, dim=2)
        re_relation_encodings, im_relation_encodings = torch.chunk(relation_encodings, 2, dim=2)
        adj_m_all = []
        # tmp = to_np(self.relations_weights_bilinear)
        # tmp0 = to_np(node_encodings)
        # tmp1 = to_np(relation_encodings)
        for relation_dim in range(self.num_relations):
            re_relation_encoding_copies = re_relation_encodings[:, relation_dim] \
                .unsqueeze(1).repeat(1, node_encodings.size()[1], 1)
            re_entity_relation_encodings = torch.cat([re_node_encodings, re_relation_encoding_copies], dim=-1)
            im_relation_encoding_copies = im_relation_encodings[:, relation_dim] \
                .unsqueeze(1).repeat(1, node_encodings.size()[1], 1)
            im_entity_relation_encodings = torch.cat([im_node_encodings, im_relation_encoding_copies], dim=-1)

            relations_weights_bilinear = self.relations_weights_bilinear[relation_dim]
            re_relations_weights_bilinear, im_relations_weights_bilinear = \
                torch.chunk(relations_weights_bilinear, 2, dim=0)
            re_relations_weights_bilinear_2d = torch.diag(re_relations_weights_bilinear)  # [dim, dim]
            im_relations_weights_bilinear_2d = torch.diag(im_relations_weights_bilinear)  # [dim, dim]

            re_score = torch.matmul(re_entity_relation_encodings, re_relations_weights_bilinear_2d) - \
                       torch.matmul(im_entity_relation_encodings, im_relations_weights_bilinear_2d)
            # tmp2 = to_np(torch.matmul(re_entity_relation_encodings, re_relations_weights_bilinear_2d))
            # tmp3 = to_np(torch.matmul(im_entity_relation_encodings, im_relations_weights_bilinear_2d))

            im_score = torch.matmul(re_entity_relation_encodings, im_relations_weights_bilinear_2d) + \
                       torch.matmul(im_entity_relation_encodings, re_relations_weights_bilinear_2d)
            # tmp4 = to_np(torch.matmul(re_entity_relation_encodings, im_relations_weights_bilinear_2d))
            # tmp5 = to_np(torch.matmul(im_entity_relation_encodings, re_relations_weights_bilinear_2d))

            # tmp6 = to_np(re_score)
            # tmp7 = to_np(im_score)
            score = torch.matmul(re_score, torch.transpose(re_entity_relation_encodings, 1, 2)) + \
                    torch.matmul(im_score, torch.transpose(im_entity_relation_encodings, 1, 2))
            # tmp8 = to_np(score)
            # m = torch.nn.Sigmoid()
            # adj_m = m(score)
            adj_m = torch.sigmoid(score)
            # tmp5 = to_np(adj_m)
            adj_m_all.append(adj_m)

        return torch.stack(adj_m_all, dim=1)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.relations_weights_bilinear.data)


class RelationalDistMult(torch.nn.Module):
    """
    implementing the DistMult for edge prediction
    reference paper: https://arxiv.org/pdf/1411.4072.pdf
    reference code: https://github.com/thunlp/OpenKE/blob/OpenKE-PyTorch/openke/module/model/DistMult.py
    """

    def __init__(self, entity_input_dim, relation_input_dim, num_relations):
        super(RelationalDistMult, self).__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.num_relations = num_relations
        self.relations_weights_bilinear = Parameter(
            torch.FloatTensor(self.num_relations, self.relation_input_dim + self.entity_input_dim))
        self.reset_parameters()

    def forward(self, node_encodings, relation_encodings):
        adj_m_all = []
        for relation_dim in range(self.num_relations):
            relation_encoding_copies = relation_encodings[:, relation_dim] \
                .unsqueeze(1).repeat(1, node_encodings.size()[1], 1)
            # [batch_size, entity_num, dim]
            entity_relation_encodings = torch.cat([node_encodings, relation_encoding_copies], dim=-1)
            relations_weights_bilinear = self.relations_weights_bilinear[relation_dim]
            relations_weights_bilinear_2d = torch.diag(relations_weights_bilinear)  # [dim, dim]
            # [batch_size, entity_num, dim] * [dim, dim]
            head_entity_relation = torch.matmul(entity_relation_encodings, relations_weights_bilinear_2d)
            # [batch_size, entity_num, dim] * [batch_size, dim, entity_num], torch.bmm() can work as well
            tail_entity_relation = torch.matmul(head_entity_relation, torch.transpose(entity_relation_encodings, 1, 2))
            adj_m = torch.sigmoid(tail_entity_relation)
            adj_m_all.append(adj_m)

        return torch.stack(adj_m_all, dim=1)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.relations_weights_bilinear.data)


class RelationalGraphConvolution(torch.nn.Module):
    """
    Simple R-GCN layer, modified from theano/keras implementation from https://github.com/tkipf/relational-gcn
    Reference Paper: https://arxiv.org/pdf/1703.06103.pdf
    We also consider relation representation here (relation labels matter)
    """

    def __init__(self, entity_input_dim, relation_input_dim, num_relations, out_dim, bias=True, num_bases=0):
        super(RelationalGraphConvolution, self).__init__()
        self.entity_input_dim = entity_input_dim
        self.relation_input_dim = relation_input_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_bases = num_bases

        if self.num_bases > 0:
            self.bottleneck_layer = torch.nn.Linear(
                (self.entity_input_dim + self.relation_input_dim) * self.num_relations, self.num_bases, bias=False)
            self.weight = torch.nn.Linear(self.num_bases, self.out_dim, bias=False)
        else:
            self.weight = torch.nn.Linear((self.entity_input_dim + self.relation_input_dim) * self.num_relations,
                                          self.out_dim, bias=False)
        if bias:
            self.bias = torch.nn.Parameter(torch.FloatTensor(self.out_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight.weight.data)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, node_features, relation_features, adj):
        # node_features: batch x num_entity x in_dim
        # relation_features: batch x num_relation x in_dim
        # adj:   batch x num_relations x num_entity x num_entity
        supports = []
        for relation_idx in range(self.num_relations):
            _r_features = relation_features[:, relation_idx: relation_idx + 1]  # batch x 1 x in_dim
            _r_features = _r_features.repeat(1, node_features.size(1), 1)  # batch x num_entity x in_dim
            supports.append(torch.bmm(adj[:, relation_idx], torch.cat([node_features, _r_features],
                                                                      dim=-1)))  # batch x num_entity x in_dim+in_dim
        supports = torch.cat(supports, dim=-1)  # batch x num_entity x (in_dim+in_dim)*num_relations
        if self.num_bases > 0:
            supports = self.bottleneck_layer(supports)
        output = self.weight(supports)  # batch x num_entity x out_dim

        if self.bias is not None:
            return output + self.bias
        else:
            return output
