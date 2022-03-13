import torch
import numpy as np
from generic.model_utils import to_pt, masked_softmax
from layers.dynamics_nn import BlocksCore, CQAttention, BlocksCoreVAE, DecoderBlock
from layers.general_nn import PointerSoftmax, EncoderBlock
from layers.graph_nn import StackedRelationalGraphConvolution, RelationalDistMult, RelationalComplEx
from layers.node_embedding import Embedding
from layers.value_nn import QNetwork


class KG_Manipulation(torch.nn.Module):
    def __init__(self, config, word_vocab, node_vocab, relation_vocab, task_name, log_file):
        """
        build the dynamic model by merging different parts together
        """
        super(KG_Manipulation, self).__init__()
        self.config = config
        self.word_vocab = word_vocab
        self.word_vocab_size = len(word_vocab)
        self.node_vocab = node_vocab
        self.node_vocab_size = len(node_vocab)
        self.relation_vocab = relation_vocab
        self.relation_vocab_size = len(relation_vocab)
        self.task_name = task_name
        self.read_config(log_file)
        self._def_layers()
        # self.print_parameters()

    def read_config(self, log_file):
        # model config
        model_config = self.config['general']['model']
        self.use_pretrained_embedding = model_config['use_pretrained_embedding']
        self.word_embedding_size = model_config['word_embedding_size']
        self.word_embedding_trainable = model_config['word_embedding_trainable']
        self.pretrained_embedding_path = "../source/embeddings/crawl-300d-2M.vec.h5"
        self.node_embedding_size = model_config['node_embedding_size']
        self.node_embedding_trainable = model_config['node_embedding_trainable']
        self.relation_embedding_size = model_config['relation_embedding_size']
        self.relation_embedding_trainable = model_config['relation_embedding_trainable']
        self.embedding_dropout = model_config['embedding_dropout']
        self.gcn_hidden_dims = model_config['gcn_hidden_dims']
        self.gcn_highway_connections = model_config['gcn_highway_connections']
        self.gcn_num_bases = model_config['gcn_num_bases']
        self.real_valued_graph = model_config['real_valued_graph']
        self.encoder_layers = model_config['encoder_layers']
        self.action_scorer_layers = model_config['action_scorer_layers']
        self.encoder_conv_num = model_config['encoder_conv_num']
        self.block_hidden_dim = model_config['block_hidden_dim']
        self.n_heads = model_config['n_heads']
        self.attention_dropout = model_config['attention_dropout']
        self.block_dropout = model_config['block_dropout']
        self.dropout = model_config['dropout']

        print("--------------------------------------------Model Settings--------------------------------------------",
              file=log_file)
        print("Encoding method: RCGNs", file=log_file)
        if 'dynamic' in self.task_name or "planning" in self.task_name:
            self.topk_num = model_config['topk_num']
            self.dynamic_model_type = model_config['dynamic_model_type']
            self.dynamic_model_mechanism = model_config['dynamic_model_mechanism']
            self.dynamic_loss_type = model_config['dynamic_loss_type']
            print("Dynamic model type: {0}".format(self.dynamic_model_type), file=log_file, flush=True)
            print("Dynamic model mechanism: {0}".format(self.dynamic_model_mechanism), file=log_file, flush=True)
            print("Dynamic loss type: {0}".format(self.dynamic_loss_type), file=log_file, flush=True)
        # if 'unsupervised' in self.task_name:
        #     print("Reward predictor: {0}".format('Atten + Linear'), file=log_file, flush=True)
        if "reward_prediction" in self.task_name or "planning" in self.task_name or 'unsupervised' in self.task_name:
            self.reward_predictor_apply_rnn = model_config['reward_predictor_apply_rnn']
            print(
                "Reward predictor: {0}".format('Atten + Rnn' if self.reward_predictor_apply_rnn else 'Atten + Linear'),
                file=log_file, flush=True)
        self.graph_decoding_method = model_config['graph_decoding_method']
        if 'dynamic' in self.task_name or 'graph_autoenc' in self.task_name or self.task_name == 'rl_planning':
            print("Decoding method: {0}".format(self.graph_decoding_method), file=log_file, flush=True)
        print("--------------------------------------------End--------------------------------------------",
              file=log_file)

    def _def_layers(self):

        # word embeddings
        if self.use_pretrained_embedding:
            self.word_embedding = Embedding(embedding_size=self.word_embedding_size,
                                            vocab_size=self.word_vocab_size,
                                            id2word=self.word_vocab,
                                            dropout_rate=self.embedding_dropout,
                                            load_pretrained=True,
                                            trainable=self.word_embedding_trainable,
                                            embedding_oov_init="random",
                                            pretrained_embedding_path=self.pretrained_embedding_path)
        else:
            self.word_embedding = Embedding(embedding_size=self.word_embedding_size,
                                            vocab_size=self.word_vocab_size,
                                            trainable=self.word_embedding_trainable,
                                            dropout_rate=self.embedding_dropout)

        # node embeddings
        self.node_embedding = Embedding(embedding_size=self.node_embedding_size,
                                        vocab_size=self.node_vocab_size,
                                        trainable=self.node_embedding_trainable,
                                        dropout_rate=self.embedding_dropout)

        # relation embeddings
        self.relation_embedding = Embedding(embedding_size=self.relation_embedding_size,
                                            vocab_size=self.relation_vocab_size,
                                            trainable=self.relation_embedding_trainable,
                                            dropout_rate=self.embedding_dropout)

        self.word_embedding_prj = torch.nn.Linear(self.word_embedding_size, self.block_hidden_dim, bias=False)

        self.rgcns = StackedRelationalGraphConvolution(
            entity_input_dim=self.node_embedding_size + self.block_hidden_dim,
            relation_input_dim=self.relation_embedding_size + self.block_hidden_dim,
            num_relations=self.relation_vocab_size,
            hidden_dims=self.gcn_hidden_dims,
            num_bases=self.gcn_num_bases,
            use_highway_connections=self.gcn_highway_connections,
            dropout_rate=self.dropout,
            real_valued_graph=self.real_valued_graph)

        if 'dynamic' in self.task_name or "planning" in self.task_name:
            if "unsupervised" in self.dynamic_loss_type:
                self.pointer_softmax = PointerSoftmax(input_dim=self.block_hidden_dim, hidden_dim=self.block_hidden_dim)
                self.obs_gen_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                     dropout=self.attention_dropout)
                self.reward_predict_action_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                                   dropout=self.attention_dropout)
                self.reward_predict_goal_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                                 dropout=self.attention_dropout)
                self.obs_gen_decoder = DecoderBlock(ch_num=self.block_hidden_dim, k=5,
                                                    block_hidden_dim=self.block_hidden_dim,
                                                    n_head=self.n_heads,
                                                    dropout=self.block_dropout)
                self.obs_gen_tgt_word_prj = torch.nn.Linear(self.block_hidden_dim, self.word_vocab_size, bias=False)
                self.goal_gen_decoder = DecoderBlock(ch_num=self.block_hidden_dim, k=5,
                                                     block_hidden_dim=self.block_hidden_dim,
                                                     n_head=self.n_heads,
                                                     dropout=self.block_dropout)
                self.goal_gen_tgt_word_prj = torch.nn.Linear(self.block_hidden_dim, self.word_vocab_size, bias=False)
                self.reward_predict_act_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim,
                                                                        bias=False)
                self.reward_predict_goal_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim,
                                                                         bias=False)
                self.reward_predict_attention_to_output = \
                    torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim * 2)
                self.reward_predict_attention_to_output_2 = \
                    torch.nn.Linear(self.block_hidden_dim * 2, self.block_hidden_dim * 1)
                self.reward_linear_predictor1 = torch.nn.Linear(self.block_hidden_dim, int(self.block_hidden_dim / 2))
                self.reward_linear_predictor2 = torch.nn.Linear(int(self.block_hidden_dim / 2), 2)
                # self.dense_adj_matrix_predictor = torch.nn.Linear(self.block_hidden_dim * self.node_vocab_size,
                #                                                   self.node_vocab_size * self.node_vocab_size)
                # We split the text encoders for different tasks
                if 'goal' in self.dynamic_loss_type:
                    self.goal_encoder = torch.nn.ModuleList(
                        [EncoderBlock(conv_num=self.encoder_conv_num,
                                      ch_num=self.block_hidden_dim, k=5,
                                      block_hidden_dim=self.block_hidden_dim,
                                      n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])
                self.reward_dropout_layer = torch.nn.Dropout(self.dropout)

                self.encoder_for_goal_gen = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num,
                                                                              ch_num=self.block_hidden_dim, k=5,
                                                                              block_hidden_dim=self.block_hidden_dim,
                                                                              n_head=self.n_heads,
                                                                              dropout=self.block_dropout) for _ in
                                                                 range(self.encoder_layers)])
                self.goal_gen_obs_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                          dropout=self.attention_dropout)
                self.goal_gen_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)

                self.encoder_for_obs_gen = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num,
                                                                             ch_num=self.block_hidden_dim, k=5,
                                                                             block_hidden_dim=self.block_hidden_dim,
                                                                             n_head=self.n_heads,
                                                                             dropout=self.block_dropout) for _ in
                                                                range(self.encoder_layers)])
                self.obs_gen_action_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                            dropout=self.attention_dropout)
                self.obs_gen_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)

        if 'dynamic' in self.task_name or "planning" in self.task_name:
            self.bce_logits_loss = torch.nn.BCEWithLogitsLoss(reduce=False)
            self.bce_loss = torch.nn.BCELoss(reduce=False)
            self.smooth_l1_loss = torch.nn.SmoothL1Loss(reduce=False)
            self.mse_loss = torch.nn.MSELoss(reduce=False)
            self.mae_loss = torch.nn.L1Loss(reduce=False)
            self.cosine_loss = torch.nn.CosineEmbeddingLoss(reduce=False)
            self.hinge_loss = torch.nn.HingeEmbeddingLoss(reduce=False)
            # We split the text encoders for different tasks
            self.action_encoder = torch.nn.ModuleList(
                [EncoderBlock(conv_num=self.encoder_conv_num,
                              ch_num=self.block_hidden_dim, k=5,
                              block_hidden_dim=self.block_hidden_dim,
                              n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])
            # We split the text encoders for different tasks
            self.obs_encoder = torch.nn.ModuleList(
                [EncoderBlock(conv_num=self.encoder_conv_num,
                              ch_num=self.block_hidden_dim, k=5,
                              block_hidden_dim=self.block_hidden_dim,
                              n_head=self.n_heads, dropout=self.block_dropout) for _ in range(self.encoder_layers)])

            if 'label' in self.dynamic_loss_type:
                self.dynamic_model = BlocksCore(action_encoding_dim=self.block_hidden_dim,
                                                node_encoding_dim=self.gcn_hidden_dims[-1],
                                                node_embedding_dim=self.node_embedding_size + self.block_hidden_dim,
                                                num_blocks_in=self.node_vocab_size, num_blocks_out=self.topk_num,
                                                n_head=self.n_heads, dropout=self.block_dropout,
                                                key_dim=self.block_hidden_dim, topk_num=self.topk_num,
                                                block_hidden_dim=self.block_hidden_dim,
                                                dynamic_loss_type=self.dynamic_loss_type,
                                                dynamic_model_type=self.dynamic_model_type,
                                                dynamic_model_mechanism=self.dynamic_model_mechanism,
                                                attention_dropout=self.attention_dropout,
                                                add_reward_flag=True if "unsupervised" in self.dynamic_loss_type else False,
                                                add_goal_flag=True if "goal" in self.dynamic_loss_type else False)
            elif 'latent' in self.dynamic_loss_type:
                self.dynamic_model = BlocksCoreVAE(action_encoding_dim=self.block_hidden_dim,
                                                   node_encoding_dim=self.gcn_hidden_dims[-1],
                                                   node_embedding_dim=self.node_embedding_size + self.block_hidden_dim,
                                                   num_blocks_in=self.node_vocab_size, num_blocks_out=1,
                                                   n_head=self.n_heads, dropout=self.block_dropout,
                                                   key_dim=self.block_hidden_dim, topk_num=self.topk_num,
                                                   block_hidden_dim=self.block_hidden_dim,
                                                   dynamic_loss_type=self.dynamic_loss_type,
                                                   dynamic_model_type=self.dynamic_model_type,
                                                   dynamic_model_mechanism=self.dynamic_model_mechanism,
                                                   attention_dropout=self.attention_dropout,
                                                   add_reward_flag=True if "unsupervised" in self.dynamic_loss_type else False,
                                                   add_goal_flag=True if "goal" in self.dynamic_loss_type else False
                                                   )
            else:
                raise ValueError("Unknown dynamic loss type {0}".format(self.dynamic_loss_type))

        if 'dynamic' in self.task_name or 'graph_autoenc' in self.task_name or self.task_name == 'rl_planning':
            if self.graph_decoding_method == 'DistMult':
                self.decode_graph = RelationalDistMult(entity_input_dim=self.gcn_hidden_dims[-1],
                                                       relation_input_dim=self.relation_embedding_size + self.block_hidden_dim,
                                                       num_relations=self.relation_vocab_size)
            elif self.graph_decoding_method == 'ComplEx':
                self.decode_graph = RelationalComplEx(entity_input_dim=self.gcn_hidden_dims[-1],
                                                      relation_input_dim=self.relation_embedding_size + self.block_hidden_dim,
                                                      num_relations=self.relation_vocab_size)
            else:
                raise ValueError("Unknown decoding method {0}".format(self.decoding_method))

        if "reward_prediction" in self.task_name or "planning" in self.task_name and 'unsupervised' not in self.task_name:
            self.reward_linear_predictor1 = torch.nn.Linear(self.block_hidden_dim, int(self.block_hidden_dim / 2))
            self.reward_linear_predictor2 = torch.nn.Linear(int(self.block_hidden_dim / 2), 2)
            self.MSE_loss = torch.nn.MSELoss(reduce=False)
            self.reward_predict_action_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim,
                                                                       bias=False)
            self.reward_predict_goal_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim,
                                                                     bias=False)
            if self.reward_predictor_apply_rnn:
                self.reward_pred_graph_rnncell = torch.nn.GRUCell(self.block_hidden_dim, self.block_hidden_dim)

            self.reward_predict_action_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                               dropout=self.attention_dropout)
            self.reward_predict_goal_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                             dropout=self.attention_dropout)
            self.reward_predict_attention_to_output = \
                torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim * 2)
            self.reward_predict_attention_to_output_2 = \
                torch.nn.Linear(self.block_hidden_dim * 2, self.block_hidden_dim * 1)

        if 'reward_prediction' in self.task_name or self.task_name == 'information_extraction' \
                or "planning" in self.task_name or 'unsupervised' in self.task_name:
            # We split the text encoders for different tasks
            self.encoder_for_reward_prediction = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num,
                                                                                   ch_num=self.block_hidden_dim, k=5,
                                                                                   block_hidden_dim=self.block_hidden_dim,
                                                                                   n_head=self.n_heads,
                                                                                   dropout=self.block_dropout) for _ in
                                                                      range(self.encoder_layers)])

        if "planning" in self.task_name:
            self.q_online_net = QNetwork(block_hidden_dim=self.block_hidden_dim,
                                         attention_dropout=self.attention_dropout,
                                         encoder_conv_num=self.encoder_conv_num,
                                         n_heads=self.n_heads,
                                         block_dropout=self.block_dropout,
                                         encoder_layers=self.encoder_layers,
                                         )
            self.q_target_net = QNetwork(block_hidden_dim=self.block_hidden_dim,
                                         attention_dropout=self.attention_dropout,
                                         encoder_conv_num=self.encoder_conv_num,
                                         n_heads=self.n_heads,
                                         block_dropout=self.block_dropout,
                                         encoder_layers=self.encoder_layers,
                                         )
            # We split the text encoders for different tasks
            self.encoder_for_q_values = torch.nn.ModuleList([EncoderBlock(conv_num=self.encoder_conv_num,
                                                                          ch_num=self.block_hidden_dim, k=5,
                                                                          block_hidden_dim=self.block_hidden_dim,
                                                                          n_head=self.n_heads,
                                                                          dropout=self.block_dropout) for _ in
                                                             range(self.encoder_layers)])

    def predict_reward_from_node_encodings(self, node_encodings):
        return self.reward_node_predictor(node_encodings)

    def encode_graph(self, node_names_word_ids, relation_names_word_ids, input_adjacency_matrices):
        # node_names_word_ids: num_node x num_word
        # relation_names_word_ids: num_relation x num_word
        # input_adjacency_matrices: batch x num_relations x num_node x num_node
        # graph node embedding / encoding
        node_embeddings = self.get_graph_embed_representations(
            node_names_word_ids, type='node')  # 1 x num_node x emb+emb
        relation_embeddings = self.get_graph_embed_representations(
            relation_names_word_ids, type='relation')  # 1 x num_relation x emb+emb

        node_embeddings = node_embeddings.repeat(
            input_adjacency_matrices.size(0), 1, 1)  # batch x num_node x emb+emb
        relation_embeddings = relation_embeddings.repeat(
            input_adjacency_matrices.size(0), 1, 1)  # batch x num_relation x emb+emb

        node_encoding_sequence = self.rgcns(node_embeddings, relation_embeddings,
                                            input_adjacency_matrices)  # batch x num_node x enc
        node_mask = torch.sum(input_adjacency_matrices[:, :-1, :, :], 1)  # batch x num_node x num_node
        node_mask = torch.sum(node_mask, -1) + torch.sum(node_mask, -2)  # batch x num_node
        node_mask = torch.gt(node_mask, 0).float()
        node_encoding_sequence = node_encoding_sequence * node_mask.unsqueeze(-1)

        return node_encoding_sequence, relation_embeddings, node_embeddings, node_mask

    def encode_dynamic_action(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        tmp1 = embeddings[0]
        squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.action_encoder[i](encoding_sequence, squared_mask,
                                                       i * (self.encoder_conv_num + 2) + 1,
                                                       self.encoder_layers)  # batch x time x enc
        tmp2 = encoding_sequence[0]
        return encoding_sequence, mask

    def encode_dynamic_obs(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.obs_encoder[i](encoding_sequence, squared_mask,
                                                    i * (self.encoder_conv_num + 2) + 1,
                                                    self.encoder_layers)  # batch x time x enc
        return encoding_sequence, mask

    def encode_dynamic_goal(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.goal_encoder[i](encoding_sequence, squared_mask,
                                                     i * (self.encoder_conv_num + 2) + 1,
                                                     self.encoder_layers)  # batch x time x enc
        return encoding_sequence, mask

    def compute_transition_dynamics(self, action_encoding_sequence,
                                    obs_encoding_sequence,
                                    action_mask, obs_mask,
                                    rewards,
                                    goal_encoding_sequence,
                                    goal_mask,
                                    node_embeddings, hx, cx,
                                    node_encodings, node_mask,
                                    ):
        """implementing the independent_recurrent_mechanism"""
        if 'label' in self.dynamic_loss_type:
            predicted_encodings, hx_new, cx_new, attn_mask = self.dynamic_model(
                act_encoding_sequence=action_encoding_sequence,
                obs_encoding_sequence=obs_encoding_sequence,
                action_mask=action_mask,
                obs_mask=obs_mask,
                rewards=rewards,
                goal_encoding_sequence=goal_encoding_sequence,
                goal_mask=goal_mask,
                node_encodings=node_encodings,
                node_embeddings=node_embeddings,
                node_mask=node_mask,
                # hx=hx, cx=cx
            )
            return predicted_encodings, hx_new, cx_new, attn_mask

        elif 'latent' in self.dynamic_loss_type:
            predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
            predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post = self.dynamic_model(
                act_encoding_sequence=action_encoding_sequence,
                obs_encoding_sequence=obs_encoding_sequence,
                action_mask=action_mask,
                obs_mask=obs_mask,
                rewards=rewards,
                goal_encoding_sequence=goal_encoding_sequence,
                goal_mask=goal_mask,
                node_encodings=node_encodings,
                node_mask=node_mask,
                node_embeddings=node_embeddings
            )

            attn_mask = None
            return predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
                   predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post, attn_mask

    def encode_text_for_reward_prediction(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len, mask for attention
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoder_for_reward_prediction[i](encoding_sequence, squared_mask,
                                                                      i * (self.encoder_conv_num + 2) + 1,
                                                                      self.encoder_layers)  # batch x time x enc
        return encoding_sequence, mask

    def encode_text_for_obs_prediction(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len, mask for attention
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoder_for_obs_gen[i](encoding_sequence, squared_mask,
                                                            i * (self.encoder_conv_num + 2) + 1,
                                                            self.encoder_layers)  # batch x time x enc
        return encoding_sequence, mask

    def encode_text_for_goal_prediction(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len, mask for attention
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoder_for_goal_gen[i](encoding_sequence, squared_mask,
                                                             i * (self.encoder_conv_num + 2) + 1,
                                                             self.encoder_layers)  # batch x time x enc
        return encoding_sequence, mask

    def encode_text_for_q_values(self, input_word_ids):
        # input_word_ids: batch x seq_len
        # text embedding / encoding
        embeddings, mask = self.embed(input_word_ids)  # batch x seq_len x emb
        squared_mask = torch.bmm(mask.unsqueeze(-1), mask.unsqueeze(1))  # batch x seq_len x seq_len, mask for attention
        encoding_sequence = embeddings
        for i in range(self.encoder_layers):
            encoding_sequence = self.encoder_for_q_values[i](encoding_sequence, squared_mask,
                                                             i * (self.encoder_conv_num + 2) + 1,
                                                             self.encoder_layers)  # batch x time x enc
        return encoding_sequence, mask

    def embed(self, input_words):
        word_embeddings, mask = self.word_embedding(input_words)  # batch x time x emb
        word_embeddings = self.word_embedding_prj(word_embeddings)
        word_embeddings = word_embeddings * mask.unsqueeze(-1)  # batch x time x hid
        return word_embeddings, mask

    def build_triple_index_mask(self, encode_type, select_index, is_cuda):
        # entity_input_dim = self.vgae_hidden_dim,
        # relation_input_dim = self.relation_embedding_size + self.block_hidden_dim
        range_number = torch.arange(select_index.size(0)).long()

        if encode_type == 'node':
            gather_index_mask = torch.zeros([select_index.size()[0], self.node_vocab_size, self.vgae_hidden_dim])
            gather_index_mask[range_number, select_index] = 1
            # tmp = torch.sum(gather_index_mask, dim=1)
        elif encode_type == 'relation':
            gather_index_mask = torch.zeros([select_index.size()[0], self.relation_vocab_size,
                                             self.relation_embedding_size + self.block_hidden_dim])
            gather_index_mask[range_number, select_index] = 1
        else:
            raise ValueError("Unknown mask type {0}".format(encode_type))
        if is_cuda:
            return gather_index_mask.type(torch.LongTensor).cuda()
        else:
            return gather_index_mask.type(torch.LongTensor)

    def bu_jiang_wu_de(self, encode_type, select_index, is_cuda, encoding):
        range_index = torch.arange(0, select_index.size(0))
        if is_cuda:
            range_index = range_index.cuda()
        if encode_type == 'node':
            select_index = range_index * self.node_vocab_size + select_index
            encoding = encoding.view([-1, self.vgae_hidden_dim])  # [batch_size*node_num, node_embed_dim]
        elif encode_type == 'relation':
            select_index = range_index * self.relation_vocab_size + select_index
            print(self.relation_embedding_size + self.block_hidden_dim)
            encoding = encoding.view([-1, self.relation_embedding_size + self.block_hidden_dim])
        else:
            raise ValueError("Unknown mask type {0}".format(encode_type))
        if is_cuda:
            select_index = select_index.type(torch.LongTensor).cuda()
        else:
            select_index = select_index.type(torch.LongTensor)
        encoding_selected = torch.index_select(input=encoding, dim=0, index=select_index)
        return encoding_selected

    def get_graph_embed_representations(self, names_word_ids, type):
        name_embeddings, _mask = self.embed(names_word_ids)  # num_node x num_word x emb
        _mask = torch.sum(_mask, -1)  # num_node
        name_embeddings = torch.sum(name_embeddings, 1)  # num_node x hid
        tmp = torch.eq(_mask, 0).float()
        if name_embeddings.is_cuda:
            tmp = tmp.cuda()
        _mask = _mask + tmp
        name_embeddings = name_embeddings / _mask.unsqueeze(-1)
        name_embeddings = name_embeddings.unsqueeze(0)  # 1 x num_node x emb

        if type == "node":
            vocab_size = self.node_vocab_size
            node_ids = np.arange(vocab_size)  # num_node
            node_ids = to_pt(node_ids, enable_cuda=names_word_ids.is_cuda, type='long').unsqueeze(0)
            # 1 x num_node
            type_embeddings, type_masks = self.node_embedding(node_ids)  # 1 x num_node x emb
        elif type == 'relation':
            vocab_size = self.relation_vocab_size
            relation_ids = np.arange(vocab_size)  # num_node
            relation_ids = to_pt(relation_ids, enable_cuda=names_word_ids.is_cuda, type='long').unsqueeze(0)
            # 1 x num_node
            type_embeddings, type_masks = self.relation_embedding(relation_ids)  # 1 x num_node x emb
        else:
            raise ValueError("Unknown embedding type {0}".format(type))

        embeddings = torch.cat([name_embeddings, type_embeddings], dim=-1)  # 1 x num_node x emb+emb
        return embeddings

    def get_subsequent_mask(self, seq):
        ''' For masking out the subsequent info. '''
        _, length = seq.size()
        subsequent_mask = torch.triu(torch.ones((length, length)), diagonal=1).float()
        subsequent_mask = 1.0 - subsequent_mask
        if seq.is_cuda:
            subsequent_mask = subsequent_mask.cuda()
        subsequent_mask = subsequent_mask.unsqueeze(0)  # 1 x time x time
        return subsequent_mask

    def decode_text(self, input_target_word_ids, h_og, obs_mask, h_go, node_mask, input_obs):
        trg_embeddings, trg_mask = self.embed(input_target_word_ids)  # batch x target_len x emb

        trg_mask_square = torch.bmm(trg_mask.unsqueeze(-1), trg_mask.unsqueeze(1))  # batch x target_len x target_len
        trg_mask_square = trg_mask_square * self.get_subsequent_mask(
            input_target_word_ids)  # batch x target_len x target_len

        obs_mask_square = torch.bmm(trg_mask.unsqueeze(-1), obs_mask.unsqueeze(1))  # batch x target_len x obs_len
        node_mask_square = torch.bmm(trg_mask.unsqueeze(-1), node_mask.unsqueeze(1))  # batch x target_len x node_len

        trg_decoder_output = trg_embeddings
        # for i in range(self.decoder_layers):
        trg_decoder_output, target_target_representations, target_source_representations, target_source_attention = \
            self.decoder[0](trg_decoder_output, trg_mask, trg_mask_square, h_og, obs_mask_square, h_go,
                            node_mask_square, 1, 1)  # batch x time x hid

        trg_decoder_output = self.tgt_word_prj(trg_decoder_output)
        trg_decoder_output = masked_softmax(trg_decoder_output, m=trg_mask.unsqueeze(-1), axis=-1)
        output = self.pointer_softmax(target_target_representations, target_source_representations, trg_decoder_output,
                                      trg_mask, target_source_attention, obs_mask, input_obs)

        return output

    def decode_for_word_gen(self, input_target_word_ids, h_item_n, word_mask, h_n_item, node_mask,
                            input_words, gen_target):
        trg_embeddings, trg_mask = self.embed(input_target_word_ids)  # batch x target_len x emb

        trg_mask_square = torch.bmm(trg_mask.unsqueeze(-1), trg_mask.unsqueeze(1))  # batch x target_len x target_len
        trg_mask_square = trg_mask_square * self.get_subsequent_mask(
            input_target_word_ids)  # batch x target_len x target_len

        word_mask_square = torch.bmm(trg_mask.unsqueeze(-1), word_mask.unsqueeze(1))
        node_mask_square = torch.bmm(trg_mask.unsqueeze(-1), node_mask.unsqueeze(1))  # batch x target_len x num_nodes

        trg_decoder_output = trg_embeddings
        if gen_target == 'obs':
            trg_decoder_output, trg_trg_representations, trg_src_representations, trg_src_attention = \
                self.obs_gen_decoder(trg_decoder_output, trg_mask, trg_mask_square, h_item_n,
                                     word_mask_square, h_n_item, node_mask_square, 1, 1)

            trg_decoder_output = self.obs_gen_tgt_word_prj(trg_decoder_output)
        elif gen_target == 'goal':
            trg_decoder_output, trg_trg_representations, trg_src_representations, trg_src_attention = \
                self.goal_gen_decoder(trg_decoder_output, trg_mask, trg_mask_square, h_item_n,
                                      word_mask_square, h_n_item, node_mask_square, 1, 1)

            trg_decoder_output = self.goal_gen_tgt_word_prj(trg_decoder_output)
        else:
            raise ValueError("Unknown gen_target {0}".format(gen_target))
        trg_decoder_output = masked_softmax(trg_decoder_output, m=trg_mask.unsqueeze(-1), axis=-1)
        # eliminating pointer softmax
        pointer_softmax_output = self.pointer_softmax(trg_trg_representations, trg_src_representations,
                                                      trg_decoder_output,
                                                      trg_mask, trg_src_attention, word_mask, input_words)
        return trg_decoder_output, pointer_softmax_output