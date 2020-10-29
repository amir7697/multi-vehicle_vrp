from torch import nn


class BertEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(BertEmbedding, self).__init__()
        self.embedding_dim = embedding_dim

        input_dimension = 3  # x, y, demand
        self.init_embedder = nn.Linear(input_dimension, self.embedding_dim)

    def forward(self, input_tensor):
        return self.init_embedder(input_tensor)
