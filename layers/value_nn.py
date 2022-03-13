import torch

from generic.model_utils import masked_mean
from layers.general_nn import CQAttention, EncoderBlock


class QNetwork(torch.nn.Module):

    def __init__(self, block_hidden_dim, attention_dropout,
                 encoder_conv_num, n_heads,
                 block_dropout, encoder_layers):
        super().__init__()

        self.block_hidden_dim = block_hidden_dim
        self.attention_dropout = attention_dropout
        self.encoder_conv_num = encoder_conv_num
        self.n_heads = n_heads
        self.block_dropout = block_dropout
        self.encoder_layers = encoder_layers

        self.q_values_action_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                     dropout=self.attention_dropout)
        self.q_values_goal_attention = CQAttention(block_hidden_dim=self.block_hidden_dim,
                                                   dropout=self.attention_dropout)
        self.q_values_action_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4,
                                                             self.block_hidden_dim,
                                                             bias=False)
        self.q_values_goal_attention_prj = torch.nn.Linear(self.block_hidden_dim * 4,
                                                           self.block_hidden_dim,
                                                           bias=False)
        self.q_values_attention_to_output = \
            torch.nn.Linear(self.block_hidden_dim * 4, self.block_hidden_dim * 2)
        self.q_values_attention_to_output_2 = \
            torch.nn.Linear(self.block_hidden_dim * 2, self.block_hidden_dim * 1)
        self.q_values_linear_predictor1 = torch.nn.Linear(self.block_hidden_dim, int(self.block_hidden_dim / 2))
        self.q_values_linear_predictor2 = torch.nn.Linear(int(self.block_hidden_dim / 2), 1)

    def forward(self, action_encoding_sequence, a_mask, goal_encoding_sequence, goal_mask, node_encodings, node_mask):

        h_ag = self.q_values_action_attention(action_encoding_sequence, node_encodings, a_mask, node_mask)
        h_ga = self.q_values_action_attention(node_encodings, action_encoding_sequence, node_mask, a_mask)
        h_ag = self.q_values_action_attention_prj(h_ag)  # bs X len X block_hidden_dim
        h_ga = self.q_values_action_attention_prj(h_ga)  # bs X len X block_hidden_dim
        ave_h_ga = masked_mean(h_ga, m=node_mask, dim=1)
        ave_h_ag = masked_mean(h_ag, m=a_mask, dim=1)

        h_lg = self.q_values_goal_attention(goal_encoding_sequence, node_encodings, goal_mask, node_mask)
        h_gl = self.q_values_goal_attention(node_encodings, goal_encoding_sequence, node_mask, goal_mask)
        h_lg = self.q_values_goal_attention_prj(h_lg)  # bs X len X block_hidden_dim
        h_gl = self.q_values_goal_attention_prj(h_gl)  # bs X len X block_hidden_dim
        ave_h_gl = masked_mean(h_gl, m=node_mask, dim=1)
        ave_h_lg = masked_mean(h_lg, m=goal_mask, dim=1)

        atten_output = self.q_values_attention_to_output(
            torch.cat([ave_h_gl, ave_h_lg, ave_h_ga, ave_h_ag], dim=1))
        atten_output = torch.tanh(atten_output)
        atten_output = self.q_values_attention_to_output_2(atten_output)
        atten_output = torch.tanh(atten_output)  # batch x block_hidden_dim

        pred_q_value = self.q_values_linear_predictor1(atten_output)
        pred_q_value = torch.tanh(pred_q_value)
        pred_q_value = self.q_values_linear_predictor2(pred_q_value)

        return pred_q_value.squeeze(-1), a_mask
