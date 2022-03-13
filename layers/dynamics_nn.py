import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from generic.model_utils import masked_softmax, PosEncoder, zero_matrix_elements, to_pt, masked_mean
from layers.general_nn import ScaledDotProductAttention, SelfAttention, BlockMultiHeadAttention, CQAttention


class RewardPredictionNodeEncoding(torch.nn.Module):

    def __init__(self, num_relations, entity_input_dim, block_hidden_dim, pred_layer=1):
        super(RewardPredictionNodeEncoding, self).__init__()
        self.pred_layer = pred_layer
        self.block_hidden_dim = block_hidden_dim
        self.num_relations = num_relations
        self.entity_input_dim = entity_input_dim
        # self.reward_weights_bilinear_layers = []
        # for i in range(self.pred_layer-1):
        #     self.reward_weights_bilinear_layers.append(Parameter(
        #         torch.FloatTensor(self.num_relations,  self.entity_input_dim, block_hidden_dim)))
        self.reward_weights_bilinear = Parameter(
            torch.FloatTensor(self.num_relations, self.entity_input_dim, int(block_hidden_dim / 2)))

        self.reward_weights_mlp1 = torch.nn.Linear(int(block_hidden_dim / 2) * self.num_relations,
                                                   self.block_hidden_dim * 2, bias=False)
        self.reward_weights_mlp2 = torch.nn.Linear(self.block_hidden_dim * 2,
                                                   2, bias=False)

        self.reset_parameters()

    def forward(self, node_encodings):
        relation_reward_output_all = []
        for relation_dim in range(self.num_relations):
            relation_reward_weights = self.reward_weights_bilinear[relation_dim]
            relation_reward_output = torch.matmul(node_encodings[:, relation_dim, :], relation_reward_weights)
            # relation_reward_output = torch.relu(relation_reward_output)
            relation_reward_output_all.append(relation_reward_output)
        reward_output0 = torch.cat(relation_reward_output_all, dim=1)
        reward_output1 = self.reward_weights_mlp1(reward_output0)
        reward_output1 = torch.tanh(reward_output1)
        reward_output = self.reward_weights_mlp2(reward_output1)

        return reward_output

    def reset_parameters(self):
        # for layer in self.reward_weights_bilinear_layers:
        torch.nn.init.xavier_uniform_(self.reward_weights_bilinear.data)
        torch.nn.init.xavier_uniform_(self.reward_weights_mlp1.weight.data)
        torch.nn.init.xavier_uniform_(self.reward_weights_mlp2.weight.data)


class BlocksCore(torch.nn.Module):
    def __init__(self, action_encoding_dim, node_encoding_dim, node_embedding_dim,
                 num_blocks_in, num_blocks_out,
                 n_head, dropout, key_dim, topk_num, block_hidden_dim,
                 dynamic_loss_type, dynamic_model_type, dynamic_model_mechanism,
                 attention_dropout, add_reward_flag=False, add_goal_flag=False):
        super().__init__()
        self.action_encoding_dim = action_encoding_dim
        self.node_encoding_dim = node_encoding_dim
        self.node_embedding_dim = node_embedding_dim
        self.num_blocks_in = num_blocks_in
        self.num_blocks_out = num_blocks_out
        self.key_dim = key_dim
        self.topk_num = topk_num
        self.block_hidden_dim = block_hidden_dim
        self.dynamic_model_type = dynamic_model_type
        self.dynamic_loss_type = dynamic_loss_type
        self.attention_dropout = attention_dropout
        self.dynamic_model_mechanism = dynamic_model_mechanism

        assert self.block_hidden_dim == self.node_encoding_dim

        self.attn_out_dim = self.action_encoding_dim
        # self.block_att = BlockMultiHeadAttention(q_input_dim=self.action_encoding_dim,
        #                                          k_input_dim=self.node_embedding_dim,
        #                                          v_input_dim=self.node_encoding_dim,
        #                                          d_model_out=self.attn_out_dim, num_blocks_read=self.num_blocks_out,
        #                                          num_blocks_write=self.num_blocks_in, d_k=self.key_dim,
        #                                          d_v=self.attn_out_dim,
        #                                          n_head=n_head, topk=num_blocks_out, grad_sparse=False, dropout=dropout)
        if 'atten' in self.dynamic_model_mechanism:
            self.block_att = SelfAttention(block_hidden_dim=block_hidden_dim, n_head=n_head)
        # self.block_att = ScaledDotProductAttention(temperature=np.power(block_hidden_dim, 2))

        if self.dynamic_model_type == 'linear':
            if add_reward_flag and not add_goal_flag:
                self.block_dynamic = BlockLinear((self.block_hidden_dim * 3 + 1) * self.num_blocks_in,
                                                 self.block_hidden_dim * self.num_blocks_in, k=self.num_blocks_in)
            elif add_goal_flag and add_reward_flag:
                self.block_dynamic = BlockLinear((self.block_hidden_dim * 4 + 1) * self.num_blocks_in,
                                                 self.block_hidden_dim * self.num_blocks_in, k=self.num_blocks_in)
            else:
                self.block_dynamic = BlockLinear((self.block_hidden_dim * 3) * self.num_blocks_in,
                                                 self.block_hidden_dim * self.num_blocks_in, k=self.num_blocks_in)
        elif self.dynamic_model_type == 'lstm':
            if add_reward_flag and not add_goal_flag:
                self.block_dynamic = BlockLSTM((self.block_hidden_dim * 3 + 1), self.block_hidden_dim,
                                               num_blocks=self.num_blocks_in)
            elif add_goal_flag and add_reward_flag:
                self.block_dynamic = BlockLSTM((self.block_hidden_dim * 4 + 1), self.block_hidden_dim,
                                               num_blocks=self.num_blocks_in)
            else:
                self.block_dynamic = BlockLSTM((self.block_hidden_dim * 3), self.block_hidden_dim,
                                               num_blocks=self.num_blocks_in)
        else:
            raise ValueError("Unknown dynamic model type {0}".format(self.dynamic_model_type))

        # self.fc_output = torch.nn.Linear(self.block_hidden_dim * self.num_blocks_in,
        #                                  self.node_encoding_dim * self.num_blocks_in, bias=False)

        self.predict_obs_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                 dropout=self.attention_dropout)
        self.obs_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
        self.predict_action_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                    dropout=self.attention_dropout)
        self.action_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
        if add_goal_flag:
            self.predict_goal_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                      dropout=self.attention_dropout)
            self.goal_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
            # self.attention_to_rnn_input = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim)

    def forward(self, act_encoding_sequence, obs_encoding_sequence, action_mask,
                obs_mask, rewards, goal_encoding_sequence, goal_mask,
                node_encodings, node_embeddings, node_mask, hx, cx):
        # action_encodings = masked_mean(act_encoding_sequence, action_mask, dim=1)
        # action_encodings = action_encodings.unsqueeze(1)
        action_encodings = act_encoding_sequence
        batch_size = action_encodings.size(0)

        attn_mask = np.ones(shape=[batch_size, self.num_blocks_in])
        attn_mask = to_pt(attn_mask, action_encodings.is_cuda)

        if 'atten' in self.dynamic_model_mechanism:
            # len_q = action_encodings.size(1)
            len_k = node_embeddings.size(1)
            input_mask = action_mask.unsqueeze(-1).repeat(1, 1, len_k)
            # input_mask = np.ones([batch_size, len_q, len_k])
            # input_mask = to_pt(input_mask, action_encodings.is_cuda)
            # assert self.num_blocks_out == len_q
            # assert self.num_blocks_in == len_k
            # node_embeddings[:, :, :32] because the first 32 dims are for node names, the last 100 dims are for node id
            attn_output, attn_weights = self.block_att(q=action_encodings,
                                                       mask=input_mask,
                                                       k=node_embeddings[:, :, :32],
                                                       v=node_encodings)
            # attn_output_reshaped = attn_output.reshape((batch_size, self.attn_out_dim * self.num_blocks_in))
            # print(attn_weights[0, 1, 71])
            # print(attn_weights[0, 2, 71])
            # print(attn_weights[0, 4, 82])
            tmp1 = torch.sum(action_mask, dim=1).unsqueeze(-1)
            tmp2 = torch.sum(attn_weights, dim=1)
            attn_weights = torch.div(torch.sum(attn_weights, dim=1), torch.sum(action_mask, dim=1).unsqueeze(-1))
            tmp = torch.sum(attn_weights, dim=1)

            attn_weights = attn_weights.unsqueeze(-1).repeat((1, 1, self.block_hidden_dim))
            # tmp = torch.topk(attn_weights, dim=1,
            #                  sorted=True, largest=False,
            #                  k=self.num_blocks_in - self.topk_num)[0]

            # topk_indices = torch.topk(attn_weights, dim=1,
            #                           sorted=True, largest=True,
            #                           k=self.topk_num)[1]
            #
            # bottomk_indices = torch.topk(attn_weights, dim=1,
            #                              sorted=True, largest=False,
            #                              k=self.num_blocks_in - self.topk_num)[1]
            #
            # # tmp1 = torch.arange(batch_size).unsqueeze(1)
            # attn_mask.index_put_(indices=(torch.arange(batch_size).unsqueeze(1), bottomk_indices),
            #                      values=torch.zeros_like(bottomk_indices[0], dtype=attn_mask.dtype))

        # attn_mask_hidden = attn_mask.reshape((batch_size, self.num_blocks_in, 1)) \
        #     .repeat((1, 1, self.block_hidden_dim))
        # attn_mask_output = attn_mask.reshape((batch_size, self.num_blocks_in, 1)). \
        #     repeat((1, 1, self.node_encoding_dim))
        # .reshape((batch_size, self.num_blocks_in * self.node_encoding_dim))

        # obs_encodings = masked_mean(obs_encoding_sequence, m=obs_mask, dim=1). \
        #     unsqueeze(1).repeat(1, node_encodings.size(1), 1)
        # action_encodings = masked_mean(act_encoding_sequence, m=action_mask, dim=1). \
        #     unsqueeze(1).repeat(1, node_encodings.size(1), 1)

        h_no = self.predict_obs_attention(node_encodings, obs_encoding_sequence, node_mask, obs_mask)
        h_no = self.obs_attention_prj(h_no)  # bs X len X block_hidden_dim
        h_na = self.predict_action_attention(node_encodings, act_encoding_sequence, node_mask, action_mask)
        h_na = self.action_attention_prj(h_na)
        if goal_encoding_sequence is not None:
            h_ng = self.predict_goal_attention(node_encodings, goal_encoding_sequence, node_mask, goal_mask)
            h_ng = self.goal_attention_prj(h_ng)

        if self.dynamic_model_type == 'linear':
            if rewards is not None and goal_encoding_sequence is None:
                rewards = rewards.unsqueeze(-1).unsqueeze(-1).repeat(1, self.num_blocks_in, 1)
                dynamic_input_mu = torch.cat([h_no, h_na, node_encodings, rewards], dim=2).reshape(
                    (batch_size, self.num_blocks_in * (self.block_hidden_dim * 3 + 1)))
            elif rewards is not None and goal_encoding_sequence is not None:
                rewards = rewards.unsqueeze(-1).unsqueeze(-1).repeat(1, self.num_blocks_in, 1)
                dynamic_input_mu = torch.cat([h_no, h_na, h_ng, node_encodings, rewards], dim=2).reshape(
                    (batch_size, self.num_blocks_in * (self.block_hidden_dim * 4 + 1)))
            else:
                dynamic_input_mu = torch.cat([h_no, h_na, node_encodings], dim=2).reshape(
                    (batch_size, self.num_blocks_in * (self.block_hidden_dim * 3)))
            hx_new = self.block_dynamic(dynamic_input_mu)
            cx_new = cx
        elif self.dynamic_model_type == 'lstm':
            # self.block_dynamic.blockify_params()
            # (batch_size, self.num_blocks_in, block_hidden_dim * 3)
            dynamic_input_mu = torch.cat([h_no, h_na, node_encodings], dim=2)
            hx_new, cx_new = self.block_dynamic(dynamic_input_mu, hx, cx)
        else:
            raise ValueError("Unknown dynamic model type {0}".format(self.dynamic_model_type))

        if hx is not None:
            # if 'atten' in self.dynamic_model_mechanism:
            #     hx_new = attn_mask_hidden * hx_new + (1 - attn_mask_hidden) * hx
            # else:
            hx_new = hx_new
        if cx is not None:
            # if 'atten' in self.dynamic_model_mechanism:
            #     cx_new = attn_mask_hidden * cx_new + (1 - attn_mask_hidden) * cx
            # else:
            cx_new = cx_new

        predicted_encodings = hx_new
        if 'atten' in self.dynamic_model_mechanism:
            predicted_encodings = attn_weights * predicted_encodings
        #     # predicted_encodings = hx_new.reshape((batch_size, self.num_blocks_in * self.block_hidden_dim))
        #     # node_encodings_flat = node_encodings.view(
        #     #     [batch_size, self.num_blocks_in * self.node_encoding_dim])
        #     predicted_encodings = attn_mask_output * hx_new + (1 - attn_mask_output) * node_encodings
        #     # predicted_encodings = predicted_encodings.reshape([batch_size, self.num_blocks_in, self.node_encoding_dim])
        # else:
        if self.dynamic_model_type == 'linear':
            return predicted_encodings, None, None, attn_mask
        elif self.dynamic_model_type == 'lstm':
            return predicted_encodings, hx_new, cx_new, attn_mask
        else:
            raise ValueError("Unknown dynamic model type {0}".format(self.dynamic_model_type))

    def blockify_params(self):
        self.block_dynamic.blockify_params()

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual


class BlocksCoreVAE(torch.nn.Module):
    def __init__(self, action_encoding_dim, node_encoding_dim, node_embedding_dim,
                 num_blocks_in, num_blocks_out,
                 n_head, dropout, key_dim, topk_num, block_hidden_dim,
                 dynamic_loss_type, dynamic_model_type, dynamic_model_mechanism, attention_dropout,
                 add_reward_flag, add_goal_flag):
        super().__init__()
        self.action_encoding_dim = action_encoding_dim
        self.node_encoding_dim = node_encoding_dim
        self.node_embedding_dim = node_embedding_dim
        self.num_blocks_in = num_blocks_in
        self.num_blocks_out = num_blocks_out
        self.key_dim = key_dim
        self.topk_num = topk_num
        self.block_hidden_dim = block_hidden_dim
        self.dynamic_model_type = dynamic_model_type
        self.dynamic_loss_type = dynamic_loss_type
        self.attention_dropout = attention_dropout
        self.dynamic_model_mechanism = dynamic_model_mechanism

        assert self.block_hidden_dim == self.node_encoding_dim

        self.attn_out_dim = self.action_encoding_dim
        self.block_att = BlockMultiHeadAttention(q_input_dim=self.action_encoding_dim,
                                                 k_input_dim=self.node_embedding_dim,
                                                 v_input_dim=self.node_encoding_dim,
                                                 d_model_out=self.attn_out_dim, num_blocks_read=self.num_blocks_out,
                                                 num_blocks_write=self.num_blocks_in, d_k=self.key_dim,
                                                 d_v=self.attn_out_dim,
                                                 n_head=n_head, topk=num_blocks_out, grad_sparse=False, dropout=dropout)
        if add_reward_flag and not add_goal_flag:
            prior_input_dim = (self.block_hidden_dim * 2) * self.num_blocks_in
            post_input_dim = (self.block_hidden_dim * 3 + 1) * self.num_blocks_in
        elif add_reward_flag and add_goal_flag:
            prior_input_dim = (self.block_hidden_dim * 3) * self.num_blocks_in
            post_input_dim = (self.block_hidden_dim * 4 + 1) * self.num_blocks_in
        else:
            prior_input_dim = (self.block_hidden_dim * 2) * self.num_blocks_in
            post_input_dim = (self.block_hidden_dim * 3) * self.num_blocks_in

        if 'single' in self.dynamic_model_mechanism:
            self.block_dynamic_mu_prior = SingleLinear(prior_input_dim,
                                                       self.block_hidden_dim * self.num_blocks_in,
                                                       k=self.num_blocks_in)
            self.block_dynamic_sigma_prior = SingleLinear(prior_input_dim,
                                                          self.block_hidden_dim * self.num_blocks_in,
                                                          k=self.num_blocks_in)
            self.block_dynamic_mu_post = SingleLinear(post_input_dim,
                                                      self.block_hidden_dim * self.num_blocks_in,
                                                      k=self.num_blocks_in)
            self.block_dynamic_sigma_post = SingleLinear(post_input_dim,
                                                         self.block_hidden_dim * self.num_blocks_in,
                                                         k=self.num_blocks_in)
        else:
            self.block_dynamic_mu_prior = BlockLinear(prior_input_dim,
                                                      self.block_hidden_dim * self.num_blocks_in,
                                                      k=self.num_blocks_in)
            self.block_dynamic_sigma_prior = BlockLinear(prior_input_dim,
                                                         self.block_hidden_dim * self.num_blocks_in,
                                                         k=self.num_blocks_in)
            self.block_dynamic_mu_post = BlockLinear(post_input_dim,
                                                     self.block_hidden_dim * self.num_blocks_in,
                                                     k=self.num_blocks_in)
            self.block_dynamic_sigma_post = BlockLinear(post_input_dim,
                                                        self.block_hidden_dim * self.num_blocks_in,
                                                        k=self.num_blocks_in)

        self.predict_obs_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                 dropout=self.attention_dropout)
        self.obs_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
        self.predict_action_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                    dropout=self.attention_dropout)
        self.action_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
        if add_goal_flag:
            self.predict_goal_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                      dropout=self.attention_dropout)
            self.goal_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim, bias=False)
        if 'atten' in self.dynamic_model_mechanism:
            self.block_att = SelfAttention(block_hidden_dim=block_hidden_dim, n_head=n_head)

    def forward(self, act_encoding_sequence, obs_encoding_sequence, action_mask,
                obs_mask, rewards, goal_encoding_sequence, goal_mask, node_encodings, node_mask, node_embeddings,
                # post_node_encodings, post_node_mask
                ):

        batch_size = act_encoding_sequence.size(0)

        if 'atten' in self.dynamic_model_mechanism:
            action_encodings = act_encoding_sequence
            len_k = node_embeddings.size(1)
            input_mask = action_mask.unsqueeze(-1).repeat(1, 1, len_k)
            attn_output, attn_weights = self.block_att(q=action_encodings,
                                                       mask=input_mask,
                                                       k=node_embeddings[:, :, :32],
                                                       v=node_encodings)
            attn_weights = torch.div(torch.sum(attn_weights, dim=1), torch.sum(action_mask, dim=1).unsqueeze(-1))
            ones = np.ones(shape=[batch_size, self.num_blocks_in])
            ones = to_pt(ones, action_encodings.is_cuda)
            attn_weights = torch.add(attn_weights, ones).unsqueeze(-1).repeat((1, 1, self.block_hidden_dim))
            node_encodings = attn_weights * node_encodings

        # action_encodings = masked_mean(act_encoding_sequence, action_mask, dim=1)
        # action_encodings = action_encodings.unsqueeze(1)  # TODO: we can compute attentions for each work in action
        # batch_size = action_encodings.size(0)

        # h_go_prior = self.predict_obs_attention(prior_node_encodings, obs_encoding_sequence, prior_node_mask, obs_mask)
        # h_go_prior = self.obs_attention_prj(h_go_prior)  # bs X len X block_hidden_dim

        if 'concatenate' in self.dynamic_model_mechanism:
            action_encodings_prior = masked_mean(act_encoding_sequence, action_mask, dim=1)
            h_na_prior = action_encodings_prior.unsqueeze(1).repeat(1, self.num_blocks_in, 1)
        else:
            h_na_prior = self.predict_action_attention(node_encodings, act_encoding_sequence, node_mask, action_mask)
            h_na_prior = self.action_attention_prj(h_na_prior)

        if goal_encoding_sequence is not None:
            if 'concatenate' in self.dynamic_model_mechanism:
                goal_encodings_prior = masked_mean(goal_encoding_sequence, goal_mask, dim=1)
                h_ng_prior = goal_encodings_prior.unsqueeze(1).repeat(1, self.num_blocks_in, 1)
            else:
                h_ng_prior = self.predict_goal_attention(node_encodings, goal_encoding_sequence, node_mask, goal_mask)
                h_ng_prior = self.goal_attention_prj(h_ng_prior)
            dynamic_input_mu_prior = torch.cat([h_na_prior, h_ng_prior, node_encodings], dim=2).reshape(
                (batch_size, self.num_blocks_in * (self.block_hidden_dim * 3)))
            dynamic_input_logvar_prior = torch.cat([h_na_prior, h_ng_prior, node_encodings], dim=2).reshape(
                (batch_size, self.num_blocks_in * (self.block_hidden_dim * 3)))
        else:
            dynamic_input_mu_prior = torch.cat([h_na_prior, node_encodings], dim=2).reshape(
                (batch_size, self.num_blocks_in * (self.block_hidden_dim * 2)))
            dynamic_input_logvar_prior = torch.cat([h_na_prior, node_encodings], dim=2).reshape(
                (batch_size, self.num_blocks_in * (self.block_hidden_dim * 2)))

        hx_new_mu_prior = self.block_dynamic_mu_prior(dynamic_input_mu_prior)
        hx_new_logvar_prior = self.block_dynamic_sigma_prior(dynamic_input_logvar_prior)
        std_prior = hx_new_logvar_prior.mul(0.5).exp_()
        eps = std_prior.data.new(std_prior.size()).normal_()
        predicted_encodings_prior = eps.mul(std_prior).add_(hx_new_mu_prior)

        if 'concatenate' in self.dynamic_model_mechanism:
            obs_encodings_post = masked_mean(obs_encoding_sequence, goal_mask, dim=1)
            h_no_post = obs_encodings_post.unsqueeze(1).repeat(1, self.num_blocks_in, 1)
            act_encoding_post = masked_mean(act_encoding_sequence, goal_mask, dim=1)
            h_na_post = act_encoding_post.unsqueeze(1).repeat(1, self.num_blocks_in, 1)
        else:
            h_no_post = self.predict_obs_attention(node_encodings, obs_encoding_sequence, node_mask, obs_mask)
            h_no_post = self.obs_attention_prj(h_no_post)  # bs X len X block_hidden_dim
            h_na_post = self.predict_action_attention(node_encodings, act_encoding_sequence, node_mask, action_mask)
            h_na_post = self.action_attention_prj(h_na_post)

        if rewards is not None and goal_encoding_sequence is None:
            rewards = rewards.unsqueeze(-1).unsqueeze(-1).repeat(1, self.num_blocks_in, 1)
            dynamic_input_mu_post = torch.cat([h_no_post, h_na_post, node_encodings, rewards], dim=2).reshape(
                (batch_size, self.num_blocks_in * (self.block_hidden_dim * 3 + 1)))
            dynamic_input_logvar_post = torch.cat([h_no_post, h_na_post, node_encodings, rewards], dim=2).reshape(
                (batch_size, self.num_blocks_in * (self.block_hidden_dim * 3 + 1)))
        elif rewards is not None and goal_encoding_sequence is not None:
            if 'concatenate' in self.dynamic_model_mechanism:
                goal_encoding_post = masked_mean(goal_encoding_sequence, goal_mask, dim=1)
                h_ng_post = goal_encoding_post.unsqueeze(1).repeat(1, self.num_blocks_in, 1)
            else:
                h_ng_post = self.predict_goal_attention(node_encodings, goal_encoding_sequence, node_mask, goal_mask)
                h_ng_post = self.goal_attention_prj(h_ng_post)

            rewards = rewards.unsqueeze(-1).unsqueeze(-1).repeat(1, self.num_blocks_in, 1)
            dynamic_input_mu_post = torch.cat([h_no_post, h_na_post, h_ng_post, node_encodings, rewards],
                                              dim=2).reshape(
                (batch_size, self.num_blocks_in * (self.block_hidden_dim * 4 + 1)))
            dynamic_input_logvar_post = torch.cat([h_no_post, h_na_post, h_ng_post, node_encodings, rewards],
                                                  dim=2).reshape(
                (batch_size, self.num_blocks_in * (self.block_hidden_dim * 4 + 1)))
        else:
            dynamic_input_mu_post = torch.cat([h_no_post, h_na_post, node_encodings], dim=2).reshape(
                (batch_size, self.num_blocks_in * (self.block_hidden_dim * 3)))
            dynamic_input_logvar_post = torch.cat([h_no_post, h_na_post, node_encodings], dim=2).reshape(
                (batch_size, self.num_blocks_in * (self.block_hidden_dim * 3)))

        hx_new_mu_post = self.block_dynamic_mu_post(dynamic_input_mu_post)
        hx_new_logvar_post = self.block_dynamic_sigma_post(dynamic_input_logvar_post)
        std_post = hx_new_logvar_post.mul(0.5).exp_()
        eps = std_post.data.new(std_post.size()).normal_()
        predicted_encodings_post = eps.mul(std_post).add_(hx_new_mu_post)

        return predicted_encodings_prior, hx_new_mu_prior, hx_new_logvar_prior, \
               predicted_encodings_post, hx_new_mu_post, hx_new_logvar_post


class SingleLinear(torch.nn.Module):
    def __init__(self, input_dim, output_dim, k):
        super(SingleLinear, self).__init__()
        assert input_dim % k == 0
        assert output_dim % k == 0
        self.sub_input = int(input_dim / k)
        self.sub_output = int(output_dim / k)

        self.k = k
        self.output_dim = output_dim
        self.input_dim = input_dim

        # self.linear_weights = Parameter(torch.FloatTensor(self.sub_input, self.sub_output))
        # self.linear_biases = Parameter(torch.FloatTensor(self.sub_output))
        self.linear_layer = torch.nn.Linear(self.sub_input, self.sub_output)
        # self.reset_parameters()

    # def reset_parameters(self):
    #     torch.nn.init.xavier_uniform_(self.linear_weights.data)
    #     torch.nn.init.zeros_(self.linear_biases.data)
        # torch.nn.init.xavier_uniform_(self.linear_biases.data)

    def forward(self, input):
        input = input.view(input.shape[0], self.k, self.sub_input)
        output_all = []
        for i in range(self.k):
            # output = torch.matmul(input[:, i, :], self.linear_weights) + self.linear_biases
            output = self.linear_layer(input[:, i, :])
            output_all.append(output)
        output_all = torch.stack(output_all, dim=1)
        return output_all


class BlockLinear(torch.nn.Module):

    def __init__(self, input_dim, output_dim, k):
        super(BlockLinear, self).__init__()
        assert input_dim % k == 0
        assert output_dim % k == 0
        self.sub_input = int(input_dim / k)
        self.sub_output = int(output_dim / k)

        self.k = k
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.linear_weights = Parameter(torch.FloatTensor(self.k, self.sub_input, self.sub_output))
        self.linear_biases = Parameter(torch.FloatTensor(self.k, self.sub_output))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.linear_weights.data)
        torch.nn.init.xavier_uniform_(self.linear_biases.data)

    def forward(self, input):
        input = input.view(input.shape[0], self.k, self.sub_input)
        output_all = []
        for i in range(self.k):
            # tmp1 = input[:, i, :]
            # tmp2 = self.linear_weights[i, :, :]
            output = torch.matmul(input[:, i, :], self.linear_weights[i, :, :]) + self.linear_biases[i, :]
            output_all.append(output)
        output_all = torch.stack(output_all, dim=1)
        return output_all
        # return output_all.view([input.shape[0], self.k * self.sub_output])

    # def blockify_params(self):
    #     pl = self.linear.parameters()
    #     for p in pl:
    #         p = p.data
    #         print(p.shape)
    #         if len(p.shape) == 2:
    #             zero_matrix_elements(p, k=self.k)


class BlockLSTM(torch.nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, input_dim, hidden_dim, num_blocks):
        super(BlockLSTM, self).__init__()

        # assert input_dim % k == 0
        # assert hidden_dim % k == 0
        # self.k = k
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.block_lstms = torch.nn.ModuleList(
            [torch.nn.LSTMCell(self.input_dim, self.hidden_dim) for i in range(self.num_blocks)])
        # self.block_lstms = torch.nn.ModuleList(
        #     [torch.nn.LSTMCell(self.input_dim, self.hidden_dim)])

    # def blockify_params(self):
    #     pl = self.lstm.parameters()
    #
    #     for p in pl:
    #         p = p.data
    #         if p.shape == torch.Size([self.hidden_dim * 4]):
    #             pass
    #             '''biases, don't need to change anything here'''
    #         if p.shape == torch.Size([self.hidden_dim * 4, self.hidden_dim]) or p.shape == torch.Size(
    #                 [self.hidden_dim * 4, self.input_dim]):
    #             for e in range(0, 4):
    #                 zero_matrix_elements(p[self.hidden_dim * e: self.hidden_dim * (e + 1)], k=self.k)

    def forward(self, input, h, c):
        hnext = []
        cnext = []
        if h is None or c is None:
            for i in range(self.num_blocks):
                lstm = self.block_lstms[i]
                # lstm = self.block_lstms[0]
                h_sub, c_sub = lstm(input[:, i, :])
                hnext.append(h_sub)
                cnext.append(c_sub)
        else:
            # h_sub_list = torch.split(h, self.num_blocks, dim=1)
            # c_sub_list = torch.split(c, self.num_blocks, dim=1)
            for i in range(self.num_blocks):
                lstm = self.block_lstms[i]
                # lstm = self.block_lstms[0]
                h_sub, c_sub = lstm(input[:, i, :], (h[:, i, :], c[:, i, :]))
                hnext.append(h_sub)
                cnext.append(c_sub)

        return torch.stack(hnext, dim=1), torch.stack(cnext, dim=1)


# class DecoderBlock(torch.nn.Module):
#     def __init__(self, ch_num, k, block_hidden_dim, n_head, dropout):
#         super().__init__()
#         self.dropout = dropout
#         self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
#         self.obs_att = SelfAttention(block_hidden_dim, n_head, dropout)
#         self.node_att = SelfAttention(block_hidden_dim, n_head, dropout)
#         self.FFN_0 = torch.nn.Linear(block_hidden_dim * 2, block_hidden_dim)
#         self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
#         self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
#         self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
#         self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)
#
# def forward(self, x, mask, self_att_mask, obs_enc_representations,
#             obs_mask, node_enc_representations, node_mask, l, blks):
#     total_layers = blks * 3
#     # conv layers
#     out = PosEncoder(x)
#     res = out
#     # self attention
#     out, _ = self.self_att(out, self_att_mask, out, out)
#     out_self = out * mask.unsqueeze(-1)
#     out = self.layer_dropout(out_self, res, self.dropout * float(l) / total_layers)
#     l += 1
#     res = out
#     out = self.norm_1(out)
#     out = F.dropout(out, p=self.dropout, training=self.training)
#     # attention with encoder outputs
#     out_obs, obs_attention = self.obs_att(out, obs_mask, obs_enc_representations, obs_enc_representations)
#     out_node, _ = self.node_att(out, node_mask, node_enc_representations, node_enc_representations)
#
#     out = torch.cat([out_obs, out_node], -1)
#     out = self.FFN_0(out)
#     out = torch.relu(out)
#     out = out * mask.unsqueeze(-1)
#
#     out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
#     l += 1
#     res = out
#     out = self.norm_2(out)
#     out = F.dropout(out, p=self.dropout, training=self.training)
#     # Fully connected layers
#     out = self.FFN_1(out)
#     out = torch.relu(out)
#     out = self.FFN_2(out)
#     out = out * mask.unsqueeze(-1)
#     out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
#     l += 1
#     return out, out_self, out_obs, obs_attention
#
#     def layer_dropout(self, inputs, residual, dropout):
#         if self.training == True:
#             pred = torch.empty(1).uniform_(0, 1) < dropout
#             if pred:
#                 return residual
#             else:
#                 return F.dropout(inputs, dropout, training=self.training) + residual
#         else:
#             return inputs + residual


class DecoderBlock(torch.nn.Module):
    def __init__(self, ch_num, k, block_hidden_dim, n_head, dropout):
        super().__init__()
        self.dropout = dropout
        self.self_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.obs_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.node_att = SelfAttention(block_hidden_dim, n_head, dropout)
        self.FFN_0 = torch.nn.Linear(block_hidden_dim * 2, block_hidden_dim)
        self.FFN_1 = torch.nn.Linear(ch_num, ch_num)
        self.FFN_2 = torch.nn.Linear(ch_num, ch_num)
        self.norm_1 = torch.nn.LayerNorm(block_hidden_dim)
        self.norm_2 = torch.nn.LayerNorm(block_hidden_dim)

    def forward(self, x, mask, self_att_mask, words_enc_representations, prev_action_mask,
                node_enc_representations, node_mask, l, blks):
        total_layers = blks * 3
        # conv layers
        out = PosEncoder(x)
        res = out
        # self attention
        out, _ = self.self_att(out, self_att_mask, out, out)
        out_self = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out_self, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_1(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # attention with encoder outputs
        out_obs, obs_attention = self.obs_att(out, prev_action_mask, words_enc_representations,
                                              words_enc_representations)
        out_node, _ = self.node_att(out, node_mask, node_enc_representations, node_enc_representations)

        out = torch.cat([out_obs, out_node], -1)
        out = self.FFN_0(out)
        out = torch.relu(out)
        out = out * mask.unsqueeze(-1)

        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        res = out
        out = self.norm_2(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        # Fully connected layers
        out = self.FFN_1(out)
        out = torch.relu(out)
        out = self.FFN_2(out)
        out = out * mask.unsqueeze(-1)
        out = self.layer_dropout(out, res, self.dropout * float(l) / total_layers)
        l += 1
        return out, out_self, out_obs, obs_attention

    def layer_dropout(self, inputs, residual, dropout):
        if self.training == True:
            pred = torch.empty(1).uniform_(0, 1) < dropout
            if pred:
                return residual
            else:
                return F.dropout(inputs, dropout, training=self.training) + residual
        else:
            return inputs + residual
