from nets.bert_embedding import BertEmbedding
from nets.bert_transformer import BertTransformer, Normalization

from itertools import permutations
from torch import nn
import torch
import random
import torch.nn.functional as F


class BertModel(nn.Module):
    def __init__(self, embedding_dim, sample_size, vehicle_capacity, device):
        super(BertModel, self).__init__()

        self.device = device
        self.embedding_dim = embedding_dim
        self.embedder = BertEmbedding(self.embedding_dim)
        self.transformer = BertTransformer(hidden_size=self.embedding_dim)
        self.sample_size = sample_size

        self.model_output_layer = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.model_output_layer_normalizer = Normalization(self.embedding_dim)
        self.vehicle_capacity = vehicle_capacity

    def forward(self, input, route):
        embedding_input = self.prepare_embedding_input(input, route)
        embedding_input = embedding_input.to(self.device)
        route = route.to(self.device)
        embedded_input = self.embedder(embedding_input)

        sample_indices = self.get_sample_indices(self.sample_size, route.size(1), route.size(0))
        sample_route = route.gather(-1, sample_indices)

        attention_mask = self.get_attention_mask(route, sample_indices)

        transformer_output = self.transformer(embedded_input, attention_mask)
        transformer_output = self.model_output_layer(transformer_output)
        transformer_output = self.model_output_layer_normalizer(transformer_output)

        sample_nodes_transformer_output = (
            transformer_output
            .gather(1, sample_indices.unsqueeze(dim=2).repeat(1, 1, transformer_output.size(2)))
        )

        sample_nodes_embedding = (
            embedded_input
            .gather(1, sample_indices.unsqueeze(dim=2).repeat(1, 1, transformer_output.size(2)))
        )

        attention_matrix = torch.matmul(sample_nodes_transformer_output,
                                        sample_nodes_embedding.transpose(-1, -2))
        log_probs = F.log_softmax(attention_matrix, dim=-1)

        likelihood = self.get_likelihood(log_probs)
        demands = self.get_demands(input)
        all_possible_routes = self.get_all_possible_routes(route, sample_indices, sample_route)
        capacity_constraint_mask = self.get_capacity_constraint_mask(all_possible_routes, demands)

        return all_possible_routes, likelihood, capacity_constraint_mask

    def get_demands(self, raw_data):
        demand_tensors_list = [sample['demand'][None, :] for sample in raw_data]
        demands = torch.cat(demand_tensors_list, dim=0).to(self.device)
        return torch.cat((-1*self.vehicle_capacity*torch.ones(demands.size(0), 1, device=self.device), demands), dim=1)

    def get_all_possible_routes(self, route, sample_indices, sample_route):
        all_indices_permutation = torch.tensor(list(permutations(list(range(self.sample_size)))), device=self.device)
        all_sample_route_permutation = sample_route[:, all_indices_permutation]
        index_permutation_augmented = (
            sample_indices
            .unsqueeze(1)
            .repeat(1, all_sample_route_permutation.size(1), 1)
        )

        route_augmented = (
            route
            .unsqueeze(1)
            .repeat(1, all_sample_route_permutation.size(1), 1)
        )

        return (
            route_augmented
            .scatter(-1, index_permutation_augmented, all_sample_route_permutation)
        )

    @staticmethod
    def get_capacity_constraint_mask(all_possible_routes, demands):
        all_possible_routes_with_depot = (
            torch.cat((
                torch.zeros((all_possible_routes.size(0), all_possible_routes.size(1), 1),
                            device=all_possible_routes.device).long()
                , all_possible_routes
            ), -1)
        )
        all_possible_capacity = (
            demands
            .unsqueeze(1).repeat(1, all_possible_routes.size(1), 1)
            .gather(-1, all_possible_routes_with_depot)
        )

        return 1 - (all_possible_capacity.cumsum(-1) > 0).any(-1).int()

    def get_likelihood(self, log_probs):
        all_indices_permutation = torch.tensor(list(permutations(list(range(self.sample_size)))), device=self.device)

        indices = (
            all_indices_permutation
            .unsqueeze(0)
            .unsqueeze(-1)
            .repeat(log_probs.size(0), 1, 1, 1)
        )
        return (
            log_probs
            .unsqueeze(1).repeat(1, indices.size(1), 1, 1)
            .gather(-1, indices)
            .sum([-1, -2])
        )

    def get_attention_mask(self, route, sample_indices):
        """
        a mask for attention that is zero where we sample and one else where
        :param route:
        :param sample_indices:
        :return: attention mask:
        """
        input_mask = (
            torch.ones(route.size(), device=self.device)
            .scatter(-1, sample_indices, torch.zeros(sample_indices.size(), device=self.device))
        )
        to_mask = input_mask.unsqueeze(1)
        broadcast_ones = torch.ones((input_mask.size(0), input_mask.size(1), 1), device=self.device)
        attention_mask = to_mask * broadcast_ones

        return attention_mask

    def get_sample_indices(self, sample_size, sample_range, num_of_samples):
        return torch.tensor([random.sample(range(sample_range), sample_size) for _ in range(num_of_samples)],
                            device=self.device)

    def prepare_embedding_input(self, raw_data, route):
        location_tensors_list = [sample['loc'][None, :, :] for sample in raw_data]
        demand_tensors_list = [sample['demand'][None, :] for sample in raw_data]
        depot_location_tensor_list = [sample['depot'][None, None, :] for sample in raw_data]

        locations = torch.cat(location_tensors_list, dim=0)
        demands = torch.cat(demand_tensors_list, dim=0)
        depot_locations = torch.cat(depot_location_tensor_list, dim=0)

        locations_with_depot = torch.cat((depot_locations, locations), dim=1)
        demands_with_depot = torch.cat((-1*self.vehicle_capacity*torch.ones(locations_with_depot.size(0), 1), demands),
                                       dim=1)

        nodes_information = torch.cat((locations_with_depot, demands_with_depot[:, :, None]), dim=-1)

        return (
            nodes_information
            .gather(1, route.unsqueeze(-1).repeat(1, 1, nodes_information.size(-1)))
        )
