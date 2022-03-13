import torch
import numpy as np
from generic.model_utils import masked_softmax, PosEncoder
import torch.nn.functional as F


class Sparse_grad_attention(torch.autograd.Function):
    # def __init__(self, top_k):
    #     super(Sparse_grad_attention,self).__init__()
    #
    #     self.sa = Sparse_attention(top_k=top_k)

    @staticmethod
    def forward(ctx, inp, sa):
        sparsified = sa(inp)
        ctx.save_for_backward(inp, sparsified)

        return inp

    @staticmethod
    def backward(ctx, grad_output):
        inp, sparsified = ctx.saved_tensors
        # print('sparsified', sparsified)
        return (grad_output) * (sparsified > 0.0).float()


class SparseAttention(torch.nn.Module):
    """
    SparseAttention from https://github.com/anirudh9119/RIMs/blob/master/event_based/sparse_attn.py
    """

    def __init__(self, top_k=5):
        super(SparseAttention, self).__init__()
        top_k += 1
        self.top_k = top_k

    def forward(self, attn_s):

        # normalize the attention weights using piece-wise Linear function
        # only top k should
        attn_plot = []
        # torch.max() returns both value and location
        # attn_s_max = torch.max(attn_s, dim = 1)[0]
        # attn_w = torch.clamp(attn_s_max, min = 0, max = attn_s_max)
        eps = 10e-8
        time_step = attn_s.size()[1]
        if time_step <= self.top_k:
            # just make everything greater than 0, and return it
            # delta = torch.min(attn_s, dim = 1)[0]
            return attn_s
        else:
            # get top k and return it
            # bottom_k = attn_s.size()[1] - self.top_k
            # value of the top k elements
            # delta = torch.kthvalue(attn_s, bottm_k, dim= 1 )[0]
            delta = torch.topk(attn_s, self.top_k, dim=1)[0][:, -1] + eps
            # delta = attn_s_max - torch.topk(attn_s, self.top_k, dim= 1)[0][:,-1] + eps
            # normalize
            delta = delta.reshape((delta.shape[0], 1))

        attn_w = attn_s - delta.repeat(1, time_step)
        attn_w = torch.clamp(attn_w, min=0)
        attn_w_sum = torch.sum(attn_w, dim=1, keepdim=True)
        attn_w_sum = attn_w_sum + eps
        attn_w_normalize = attn_w / attn_w_sum.repeat(1, time_step)

        # print('attn', attn_w_normalize)

        return attn_w_normalize


class ScaledDotProductAttention(torch.nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, topk=None, grad_sparse=False, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=2)
        self.grad_sparse = grad_sparse
        self.topk = topk
        if self.topk is not None:
            self.sa = SparseAttention(top_k=topk)  # k=2
        # self.sga = Sparse_grad_attention(top_k=2)
        self.dropout = torch.nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask, use_sparse=False):
        attn = torch.bmm(q, k.transpose(1, 2))  # [batch_size, len, dim] x [batch_size, dim, len]
        attn = attn / self.temperature
        # if mask is not None:
        #     attn = attn.masked_fill(mask, -np.inf)
        #     # attn = self.dropout(attn)
        # attn = self.softmax(attn)
        attn = masked_softmax(attn, mask, 2)
        # use_sparse = True  # False
        if use_sparse:
            mb, ins, outs = attn.shape[0], attn.shape[1], attn.shape[2]
            sparse_attn = attn.reshape((mb * ins, outs))
            # print('sparse attn shape 1', sparse_attn.shape)
            # sga = Sparse_grad_attention(2)
            if self.grad_sparse:
                sga = Sparse_grad_attention(self.topk)
                sparse_attn = sga(sparse_attn)
            else:
                sparse_attn = self.sa(sparse_attn)
            sparse_attn = sparse_attn.reshape((mb, ins, outs))
            attn = sparse_attn * 1.0

        attn = self.dropout(attn)
        # tmp = torch.sum(attn, dim=2)
        output = torch.bmm(attn, v)

        return output, attn


class SelfAttention(torch.nn.Module):
    ''' From Multi-Head Attention module
    https://github.com/jadore801120/attention-is-all-you-need-pytorch'''

    def __init__(self, block_hidden_dim, n_head, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.block_hidden_dim = block_hidden_dim
        self.w_qs = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_ks = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        self.w_vs = torch.nn.Linear(block_hidden_dim, n_head * block_hidden_dim, bias=False)
        torch.nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        torch.nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (block_hidden_dim * 2)))
        self.attention = ScaledDotProductAttention(temperature=np.power(block_hidden_dim, 0.5))
        self.fc = torch.nn.Linear(n_head * block_hidden_dim, block_hidden_dim)
        self.layer_norm = torch.nn.LayerNorm(self.block_hidden_dim)
        torch.nn.init.xavier_normal_(self.fc.weight)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, mask, k, v):
        # q: batch x len_q x hid
        # k: batch x len_k x hid
        # v: batch x len_v x hid
        # mask: batch x len_q x len_k
        batch_size, len_q = q.size(0), q.size(1)
        len_k, len_v = k.size(1), v.size(1)
        assert mask.size(1) == len_q
        assert mask.size(2) == len_k
        residual = q

        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.block_hidden_dim)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.block_hidden_dim)
        v = self.w_vs(v).view(batch_size, len_v, self.n_head, self.block_hidden_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.block_hidden_dim)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.block_hidden_dim)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.block_hidden_dim)  # (n*b) x lv x dv

        mask = mask.repeat(self.n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(self.n_head, batch_size, len_q, self.block_hidden_dim)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class BlockMultiHeadAttention(torch.nn.Module):
    """From Recurrent Independent Mechanisms.
    https://github.com/anirudh9119/RIMs/blob/master/event_based/attention.py
    """

    def __init__(self, q_input_dim, k_input_dim, v_input_dim, d_model_out,
                 num_blocks_read, num_blocks_write, d_k, d_v, n_head,
                 topk, grad_sparse, residual=False, dropout=0.1, skip_write=False):
        super().__init__()
        self.q_input_dim = q_input_dim
        self.k_input_dim = k_input_dim
        self.v_input_dim = v_input_dim
        self.d_model_out = d_model_out
        self.num_blocks_read = num_blocks_read
        self.num_blocks_write = num_blocks_write
        self.k_dim = d_k
        self.v_dim = d_v
        self.n_head = n_head
        self.residual = residual

        self.w_qs = GroupLinearLayer(self.q_input_dim, self.n_head * self.k_dim, self.num_blocks_read)
        self.w_qs.reset_parameters()
        self.w_ks = GroupLinearLayer(self.k_input_dim, self.n_head * self.k_dim, self.num_blocks_write)
        self.w_ks.reset_parameters()
        self.w_vs = GroupLinearLayer(self.v_input_dim, self.n_head * self.v_dim, self.num_blocks_write)
        self.w_vs.reset_parameters()

        self.attention = ScaledDotProductAttention(temperature=np.power(self.k_dim, 0.5),
                                                   topk=topk,
                                                   grad_sparse=grad_sparse)

        self.gate_fc = torch.nn.Linear(self.n_head * self.v_dim, self.d_model_out)
        torch.nn.init.xavier_normal_(self.gate_fc.weight)

        if not skip_write:
            self.fc = torch.nn.Linear(self.n_head * self.v_dim, self.d_model_out)
            torch.nn.init.xavier_normal_(self.fc.weight)
        else:
            self.fc = lambda a: a

        self.layer_norm = torch.nn.LayerNorm(self.d_model_out)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, q, mask, k, v):
        # q: batch x len_q x hid
        # k: batch x len_k x hid
        # v: batch x len_v x hid
        # mask: batch x len_q x len_k
        batch_size, len_q = q.size(0), q.size(1)
        len_k, len_v = k.size(1), v.size(1)
        assert mask.size(1) == len_q
        assert mask.size(2) == len_k
        residual = v

        q = self.w_qs(q).view(batch_size, len_q, self.n_head, self.k_dim)
        k = self.w_ks(k).view(batch_size, len_k, self.n_head, self.k_dim)
        v = self.w_vs(v).view(batch_size, len_v, self.n_head, self.v_dim)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, self.k_dim)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, self.k_dim)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, self.v_dim)  # (n*b) x lv x dv

        mask = mask.repeat(self.n_head, 1, 1)  # (n*b) x .. x ..
        output, attn = self.attention(q, k, v, mask=mask)

        output = output.view(self.n_head, batch_size, len_q, self.v_dim)
        output = output.permute(1, 2, 0, 3).contiguous().view(batch_size, len_q, -1)  # b x lq x (n*dv)
        output = self.dropout(self.fc(output))

        # TODO: Maybe consider different kinds of residuals
        # gate = torch.sigmoid(self.gate_fc(output))
        # if self.residual:
        #     output = gate * torch.tanh(output)
        # else:
        #     pass

        output = self.layer_norm(output + residual)
        return output, attn


class GroupLinearLayer(torch.nn.Module):
    """From Recurrent Independent Mechanisms.
    https://github.com/anirudh9119/RIMs/blob/master/event_based/GroupLinearLayer.py
    """

    def __init__(self, input_dim, output_dim, num_blocks):
        super(GroupLinearLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(num_blocks, self.input_dim, self.output_dim))
        # self.weight = torch.nn.Parameter(0.01 * torch.randn(num_blocks, din, dout))

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = torch.bmm(x, self.weight)
        return x.permute(1, 0, 2)

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight.data)
        torch.nn.init.normal_(self.weight.data, mean=0, std=np.sqrt(2.0 / (self.input_dim * 2)))


class CQAttention(torch.nn.Module):
    """
    check paper Bidirectional Attention Flow for Machine Comprehension

    """

    def __init__(self, block_hidden_dim, dropout=0):
        super().__init__()
        self.dropout = dropout
        w4C = torch.empty(block_hidden_dim, 1)
        w4Q = torch.empty(block_hidden_dim, 1)
        w4mlu = torch.empty(1, 1, block_hidden_dim)
        torch.nn.init.xavier_uniform_(w4C)
        torch.nn.init.xavier_uniform_(w4Q)
        torch.nn.init.xavier_uniform_(w4mlu)
        self.w4C = torch.nn.Parameter(w4C)
        self.w4Q = torch.nn.Parameter(w4Q)
        self.w4mlu = torch.nn.Parameter(w4mlu)

        bias = torch.empty(1)
        torch.nn.init.constant_(bias, 0)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, C, Q, Cmask, Qmask):
        S = self.trilinear_for_attention(C, Q)
        Cmask = Cmask.unsqueeze(-1)
        Qmask = Qmask.unsqueeze(1)
        S1 = masked_softmax(S, Qmask, axis=2)
        S2 = masked_softmax(S, Cmask, axis=1)
        A = torch.bmm(S1, Q)
        B = torch.bmm(torch.bmm(S1, S2.transpose(1, 2)), C)
        out = torch.cat([C, A, torch.mul(C, A), torch.mul(C, B)], dim=2)
        return out

    def trilinear_for_attention(self, C, Q):
        C = F.dropout(C, p=self.dropout, training=self.training)
        Q = F.dropout(Q, p=self.dropout, training=self.training)
        max_q_len = Q.size(-2)
        max_context_len = C.size(-2)
        subres0 = torch.matmul(C, self.w4C).expand([-1, -1, max_q_len])
        subres1 = torch.matmul(Q, self.w4Q).transpose(1, 2).expand([-1, max_context_len, -1])
        subres2 = torch.matmul(C * self.w4mlu, Q.transpose(1, 2))
        res = subres0 + subres1 + subres2
        res += self.bias
        return res


class PointerSoftmax(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.pointer_softmax_context = torch.nn.Linear(input_dim, hidden_dim)
        self.pointer_softmax_target = torch.nn.Linear(input_dim, hidden_dim)
        self.pointer_softmax_squash = torch.nn.Linear(hidden_dim, 1)

    def forward(self, target_target_representations, target_source_representations, trg_decoder_output, target_mask,
                target_source_attention, source_mask, input_source):
        # target_target_representations:    batch x target_len x hid
        # target_source_representations:    batch x target_len x hid
        # trg_decoder_output:               batch x target len x vocab
        # target mask:                      batch x target len
        # target_source_attention:          batch x target len x source len
        # source mask:                      batch x source len
        # input source:                     batch x source len
        batch_size = target_source_attention.size(0)
        target_len = target_source_attention.size(1)
        source_len = target_source_attention.size(2)

        switch = self.pointer_softmax_context(target_source_representations)  # batch x trg_len x hid
        switch = switch + self.pointer_softmax_target(target_target_representations)  # batch x trg_len x hid
        switch = torch.tanh(switch)
        switch = switch * target_mask.unsqueeze(-1)
        switch = self.pointer_softmax_squash(switch).squeeze(-1)  # batch x trg_len
        switch = torch.sigmoid(switch)
        switch = switch * target_mask  # batch x target len
        switch = switch.unsqueeze(-1)  # batch x target len x 1

        target_source_attention = target_source_attention * source_mask.unsqueeze(1)
        from_vocab = trg_decoder_output  # batch x target len x vocab
        from_source = torch.autograd.Variable(
            torch.zeros(batch_size * target_len, from_vocab.size(-1)))  # batch x target len x vocab
        if from_vocab.is_cuda:
            from_source = from_source.cuda()
        input_source = input_source.unsqueeze(1).expand(batch_size, target_len, source_len)
        input_source = input_source.contiguous().view(batch_size * target_len, -1)  # batch*target_len x source_len
        from_source = from_source.scatter_add_(1, input_source,
                                               target_source_attention.view(batch_size * target_len, -1))
        from_source = from_source.view(batch_size, target_len, -1)  # batch x target_len x vocab
        merged = switch * from_vocab + (1.0 - switch) * from_source  # batch x target_len x vocab
        merged = merged * target_mask.unsqueeze(-1)
        return merged


class EncoderBlock(torch.nn.Module):
    def __init__(self, conv_num, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.convs = torch.nn.ModuleList([DepthwiseSeparableConv(ch_num, ch_num, k) for _ in range(conv_num)])
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
        self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
        self.norm_C = torch.nn.ModuleList([torch.nn.LayerNorm(block_hidden_dim) for _ in range(conv_num)])
        self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)
        self.conv_num = conv_num

    def forward(self, x, mask, l, blks):
        total_layers = (self.conv_num + 2) * blks
        # conv layers
        out = PosEncoder(x)
        for i, conv in enumerate(self.convs):
            res = out
            out = self.norm_C[i](out)
            if (i) % 2 == 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            out = conv(out)
            out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
            l += 1
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # self attention
        out, _ = self.self_att(out, mask, out, out)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # fully connected layers
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        return out

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class DepthwiseSeparableConv(torch.nn.Module):
    def __init__(self, in_ch, out_ch, k, bias=True):
        super().__init__()
        self.depthwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=in_ch,
                                              kernel_size=k, groups=in_ch, padding=k // 2, bias=False)
        self.pointwise_conv = torch.nn.Conv1d(in_channels=in_ch, out_channels=out_ch,
                                              kernel_size=1, padding=0, bias=bias)

    def forward(self, x):
        x = x.transpose(1, 2)
        res = torch.relu(self.pointwise_conv(self.depthwise_conv(x)))
        res = res.transpose(1, 2)
        return res
