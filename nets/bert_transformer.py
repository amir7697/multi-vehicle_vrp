import torch
import math
import torch.nn.functional as F

from torch import nn


class BertTransformer(nn.Module):
    def __init__(self,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_dropout_prob=0.1,
                 hidden_dropout_prob=0.2
                 ):
        super(BertTransformer, self).__init__()
        self.attention_head_size = hidden_size // num_attention_heads
        self.num_attention_head = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

        self.query_layer = nn.Linear(hidden_size, hidden_size)
        self.key_layer = nn.Linear(hidden_size, hidden_size)
        self.value_layer = nn.Linear(hidden_size, hidden_size)

        self.attention_dropout_prob = attention_dropout_prob

        self.attention_projection_list = [nn.Linear(hidden_size, hidden_size) for _ in range(self.num_hidden_layers)]
        self.hidden_dropout_prob = hidden_dropout_prob

        self.intermediate_normalizer = Normalization(hidden_size)
        self.intermediate_feed_forward_layer = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.ReLU()
        )
        self.output_feed_forward_layer = nn.Linear(intermediate_size, hidden_size)
        self.output_normalizer = Normalization(hidden_size)

    def forward(self, input, attention_mask):
        prev_output = input
        for i in range(self.num_hidden_layers):
            layer_input = prev_output
            attention_head = self.attention_layer(layer_input, layer_input, attention_mask)
            projection = self.attention_projection_list[i].to(layer_input.device)
            attention_output = projection(attention_head)
            F.dropout(attention_output, self.hidden_dropout_prob, training=True, inplace=True)

            attention_output = self.intermediate_normalizer(attention_output + layer_input)
            intermediate_layer = self.intermediate_feed_forward_layer(attention_output)
            layer_output = self.output_feed_forward_layer(intermediate_layer)
            F.dropout(layer_output, self.hidden_dropout_prob, training=True, inplace=True)
            layer_output = self.output_normalizer(layer_output + attention_output)
            prev_output = layer_output

        return prev_output

    def attention_layer(self,
                        from_tensor,
                        to_tensor,
                        attention_mask=None,
                        query_activation=True,
                        key_activation=True,
                        value_activation=True,
                        ):
        """
        Performs multi-headed attention from `from_tensor` to `to_tensor`.

        If `from_tensor` and `to_tensor` are the same, then this is self-attention.

        `from_tensor` ==> query
        `to_tensor` ==> key, value
        """
        batch_size, from_number_of_node, hidden_dim = from_tensor.shape
        # query: batch size * from nodes * hidden dim
        query = self.query_layer(from_tensor)
        if query_activation:
            F.relu_(query)
        # key: batch size * to tensor * hidden dim
        key = self.key_layer(to_tensor)
        if key_activation:
            F.relu_(key)
        # value: batch size * to tensor * hidden dim
        value = self.value_layer(to_tensor)
        if value_activation:
            F.relu_(value)

        # query layer: batch size * num attention head * from tensor * attention head size
        query_layer = self.make_heads(query)
        # key layer: batch size * num attention head * to tensor * attention head size
        key_layer = self.make_heads(key)

        # attention score: batch size * num attention head * from tensor * to tensor
        attention_score = torch.matmul(query_layer, key_layer.transpose(-2, -1)) / math.sqrt(query_layer.size(-1))
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)

            # add negative value to masked positions
            attention_score += (1 - attention_mask)*-100000

        attention_probs = torch.softmax(attention_score, dim=-1)
        attention_probs = F.dropout(attention_probs, self.attention_dropout_prob, training=True)

        # value layer: batch size * num attention head * to tensor * attention head size
        value_layer = self.make_heads(value)
        # context layer: batch size * num attention head * from tensor * attention head size
        context = (
            torch.matmul(attention_probs, value_layer)
            .transpose(1, 2)
            .reshape(batch_size, from_number_of_node, hidden_dim)
        )

        return context

    def make_heads(self, input_tensor):
        batch_size, number_of_nodes, hidden_dim = input_tensor.size()
        return (
            input_tensor
            .reshape(batch_size, number_of_nodes, self.num_attention_head, -1)
            .transpose(1, 2)
        )


class Normalization(nn.Module):
    def __init__(self, embed_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.BatchNorm1d(embed_dim, affine=True)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())

