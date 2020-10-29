import torch
import random


def get_sample_indices(batch_size, sequence_length, sample_size):
    return torch.tensor([random.sample(range(sequence_length), sample_size) for _ in range(batch_size)])
