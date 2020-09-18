import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math
from typing import NamedTuple

from nets.graph_encoder import GraphAttentionEncoder
from torch.nn import DataParallel


def set_decode_type(model, decode_type):
    if isinstance(model, DataParallel):
        model = model.module
    model.set_decode_type(decode_type)


class AttentionModelFixed(NamedTuple):
    """
    Context for AttentionModel decoder that is fixed during decoding so can be precomputed/cached
    This class allows for efficient indexing of multiple Tensors at once
    """
    node_embeddings: torch.Tensor
    context_node_projected: torch.Tensor
    glimpse_key: torch.Tensor
    glimpse_val: torch.Tensor
    logit_key: torch.Tensor

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):
            return AttentionModelFixed(
                node_embeddings=self.node_embeddings[key],
                context_node_projected=self.context_node_projected[key],
                glimpse_key=self.glimpse_key[:, key],  # dim 0 are the heads
                glimpse_val=self.glimpse_val[:, key],  # dim 0 are the heads
                logit_key=self.logit_key[key]
            )
        return super(AttentionModelFixed, self).__getitem__(key)


class AttentionModel(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 problem,
                 graph_size,
                 distance_embedding,
                 cost_coefficients,
                 vehicle_count=1,
                 n_encode_layers=2,
                 tanh_clipping=10.,
                 mask_inner=True,
                 mask_logits=True,
                 normalization='batch',
                 n_heads=8,
                 checkpoint_encoder=False,
                 shrink_size=None):
        super(AttentionModel, self).__init__()

        self.cost_coefficients = cost_coefficients
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_encode_layers = n_encode_layers
        self.decode_type = None
        self.temp = 1.0
        self.is_vrp = problem.NAME == 'cvrp'
        self.is_vrptw = problem.NAME == 'cvrptw'

        self.tanh_clipping = tanh_clipping

        self.mask_inner = mask_inner
        self.mask_logits = mask_logits

        self.problem = problem
        self.n_heads = n_heads
        self.checkpoint_encoder = checkpoint_encoder
        self.shrink_size = shrink_size
        self.vehicle_count = vehicle_count
        self.distance_embedding = distance_embedding

        if self.is_vrp:
            if self.distance_embedding:
                node_dim_depot = graph_size # distance to each node
                node_dim = graph_size + 1 # distance to each node, demand
            else:
                node_dim_depot = 2 # x, y
                node_dim = 3 # x, y, demand
            # Embedding of last node + remaining_capacity  per vehicle
            step_context_dim = (embedding_dim + 1) * self.vehicle_count
        elif self.is_vrptw:
            if self.distance_embedding:
                node_dim_depot = graph_size # distance to each node
                node_dim = graph_size + 3 # distance to each node, demand, start time, finish time
            else:
                node_dim_depot = 2 # x, y
                node_dim = 5 # x, y, demand, start time, finish time
            # Embedding of last node + remaining_capacity + current time per vehicle
            step_context_dim = (embedding_dim + 2) * self.vehicle_count

        # Special embedding projection for depot node. the depot does not have demand
        self.init_embed_depot = nn.Linear(node_dim_depot, embedding_dim)
        self.init_embed = nn.Linear(node_dim, embedding_dim)

        self.embedder = GraphAttentionEncoder(
            n_heads=n_heads,
            embed_dim=embedding_dim,
            n_layers=self.n_encode_layers,
            normalization=normalization
        )

        # For each node we compute (glimpse key, glimpse value, logit key) so 3 * embedding_dim
        self.project_node_embeddings = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)
        self.project_fixed_context = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.project_step_context = nn.Linear(step_context_dim, embedding_dim, bias=False)
        assert embedding_dim % n_heads == 0
        # Note n_heads * val_dim == embedding_dim so input to project_out is embedding_dim
        self.project_out = nn.Linear(embedding_dim, embedding_dim, bias=False)

    def set_decode_type(self, decode_type, temp=None):
        self.decode_type = decode_type
        if temp is not None:  # Do not change temperature if not provided
            self.temp = temp

    def forward(self, input, return_pi=False, return_cost_detail=False):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :param return_pi: whether to return the output sequences, this is optional as it is not compatible with
        using DataParallel as the results may be of different lengths on different GPUs
        :return:
        """

        if self.checkpoint_encoder and self.training:  # Only checkpoint if we need gradients
            embeddings, _ = checkpoint(self.embedder, self._init_embed(input))
        else:
            embeddings, _ = self.embedder(self._init_embed(input))

        _log_p, pi = self._inner(input, embeddings)

        cost, mask, distance_cost, early_cost, delay_cost = self.problem.get_costs(
            dataset=input, pi=pi, cost_coefficients=self.cost_coefficients, vehicle_count=self.vehicle_count)
        # Log likelyhood is calculated within the model since returning it per action does not work well with
        # DataParallel since sequences can be of different lengths
        ll = self._calc_log_likelihood(_log_p, pi, mask)
        if return_cost_detail:
            if return_pi:
                return cost, ll, pi, distance_cost, early_cost, delay_cost

            return cost, ll, distance_cost, early_cost, delay_cost
        else:
            if return_pi:
                return cost, ll, pi

            return cost, ll

    def _calc_log_likelihood(self, _log_p, a, mask):
        # Get log_p corresponding to selected actions
        log_p = _log_p.gather(2, a.unsqueeze(-1)).squeeze(-1)

        # Optional: mask out actions irrelevant to objective so they do not get reinforced
        if mask is not None:
            log_p[mask] = 0

        assert (log_p > -1000).data.all(), "Logprobs should not be -inf, check sampling procedure!"

        # Calculate log_likelihood
        return log_p.sum(1)

    def _init_embed(self, input):
        if self.is_vrp:
            features = ('demand',)
        elif self.is_vrptw:
            features = ('demand', 'timeWindowStart', 'timeWindowFinish')

        if self.distance_embedding:
            distance_to_depot = self._calculate_distance_to_depot(input['depot'], input['loc'])
            distance_matrix = self._calculate_distance_matrix(input['loc'])

            return torch.cat(
                (
                    self.init_embed_depot(distance_to_depot)[:, None, :],
                    self.init_embed(torch.cat((
                        distance_matrix,
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ), 1
            )
        else:
            return torch.cat(
                (
                    self.init_embed_depot(input['depot'])[:, None, :],
                    self.init_embed(torch.cat((
                        input['loc'],
                        *(input[feat][:, :, None] for feat in features)
                    ), -1))
                ),
                1
            )

    def _inner(self, input, embeddings):

        outputs = []
        sequences = []

        state = self.problem.make_state(input, vehicle_count=self.vehicle_count)

        # Compute keys, values for the glimpse and keys for the logits once as they can be reused in every step
        fixed = self._precompute(embeddings)

        while not state.all_finished():
            log_p, mask = self._get_log_p(fixed, state)
            selected = self._select_node(log_p.exp()[:, 0, :], mask[:, 0, :])  # Squeeze out steps dimension
            state = state.update(selected, self.vehicle_count)
            outputs.append(log_p[:, 0, :])
            sequences.append(selected)

        # Collected lists, return Tensor
        return torch.stack(outputs, 1), torch.stack(sequences, 1)

    def _select_node(self, probs, mask):

        assert (probs == probs).all(), "Probs should not contain any nans"

        if self.decode_type == "greedy":
            _, selected = probs.max(1)
            assert not mask.gather(1, selected.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"

        elif self.decode_type == "sampling":
            selected = probs.multinomial(1).squeeze(1)

            # Check if sampling went OK, can go wrong due to bug on GPU
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            while mask.gather(1, selected.unsqueeze(-1)).data.any():
                print('Sampled bad values, resampling!')
                selected = probs.multinomial(1).squeeze(1)

        else:
            assert False, "Unknown decode type"
        return selected

    def _precompute(self, embeddings, num_steps=1):

        # The fixed context projection of the graph embedding is calculated only once for efficiency
        graph_embed = embeddings.mean(1)
        # fixed context = (batch_size, 1, embed_dim) to make broadcastable with parallel timesteps
        fixed_context = self.project_fixed_context(graph_embed)[:, None, :]

        multi_vehicle_embedding = (
            embeddings[:, :, None, :]
            .repeat(1, 1, self.vehicle_count, 1)
            .view(embeddings.shape[0], self.vehicle_count * embeddings.shape[1], embeddings.shape[2])
        )
        glimpse_key_fixed, glimpse_val_fixed, logit_key_fixed = \
            self.project_node_embeddings(multi_vehicle_embedding[:, None, :, :]).chunk(3, dim=-1)

        # No need to rearrange key for logit as there is a single head
        fixed_attention_node_data = (
            self._make_heads(glimpse_key_fixed, num_steps),
            self._make_heads(glimpse_val_fixed, num_steps),
            logit_key_fixed.contiguous()
        )
        return AttentionModelFixed(embeddings, fixed_context, *fixed_attention_node_data)

    def _get_log_p(self, fixed, state, normalize=True):

        # Compute query = context node embedding
        query = fixed.context_node_projected + \
                self.project_step_context(self._get_parallel_step_context(fixed.node_embeddings, state))

        # Compute keys and values for the nodes
        glimpse_K, glimpse_V, logit_K = fixed.glimpse_key, fixed.glimpse_val, fixed.logit_key

        # Compute the mask
        mask = state.get_mask(self.vehicle_count)

        # Compute logits (unnormalized log_p)
        log_p, glimpse = self._one_to_many_logits(query, glimpse_K, glimpse_V, logit_K, mask)

        if normalize:
            log_p = torch.log_softmax(log_p / self.temp, dim=-1)

        assert not torch.isnan(log_p).any()

        return log_p, mask

    def _get_parallel_step_context(self, embeddings, state, from_depot=False):
        """
        Returns the context per step, optionally for multiple steps at once (for efficient evaluation of the model)

        :param embeddings: (batch_size, graph_size, embed_dim)
        :param prev_a: (batch_size, num_steps)
        :param first_a: Only used when num_steps = 1, action of first step or None if first step
        :return: (batch_size, num_steps, context_dim)
        """

        current_node = state.get_current_node()
        batch_size, num_steps, vehicle_count = current_node.size()

        if self.is_vrp:
            if from_depot:
                return torch.cat(
                    (
                        embeddings[:, 0:1, :]
                        .view(batch_size, num_steps, 1, embeddings.size(-1))
                        .expand(batch_size, num_steps, vehicle_count, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, :, None])
                    ),
                    -1
                ).view(batch_size, num_steps, -1)
            else:
                return torch.cat(
                    (
                        torch.gather(
                            embeddings[:, :, None, :].repeat(1, 1, vehicle_count, 1),
                            1,
                            current_node.contiguous()
                            .view(batch_size, num_steps, vehicle_count, 1)
                            .repeat(1, 1, 1, embeddings.size(-1))
                        ).view(batch_size, num_steps, vehicle_count, embeddings.size(-1)),
                        self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, :, None]
                    ),
                    -1
                ).view(batch_size, num_steps, -1)
        elif self.is_vrptw:
            if from_depot:
                return torch.cat(
                    (
                        embeddings[:, 0:1, :]
                        .view(batch_size, num_steps, 1, embeddings.size(-1))
                        .expand(batch_size, num_steps, vehicle_count, embeddings.size(-1)),
                        # used capacity is 0 after visiting depot
                        self.problem.VEHICLE_CAPACITY - torch.zeros_like(state.used_capacity[:, :, :, None]),
                        torch.zeros_like(state.cur_time[:, None, :, None])
                    ),
                    -1
                ).view(batch_size, num_steps, -1)
            else:
                return torch.cat(
                    (
                        torch.gather(
                            embeddings[:, :, None, :].repeat(1, 1, vehicle_count, 1),
                            1,
                            current_node.contiguous()
                            .view(batch_size, num_steps, vehicle_count, 1)
                            .repeat(1, 1, 1, embeddings.size(-1))
                        ).view(batch_size, num_steps, vehicle_count, embeddings.size(-1)),
                        self.problem.VEHICLE_CAPACITY - state.used_capacity[:, :, :, None],
                        state.cur_time[:, None, :, None]
                    ),
                    -1
                ).view(batch_size, num_steps, -1)

    def _one_to_many_logits(self, query, glimpse_K, glimpse_V, logit_K, mask):

        batch_size, num_steps, embed_dim = query.size()
        key_size = val_size = embed_dim // self.n_heads

        # Compute the glimpse, rearrange dimensions so the dimensions are (n_heads, batch_size, num_steps, 1, key_size)
        glimpse_Q = query.view(batch_size, num_steps, self.n_heads, 1, key_size).permute(2, 0, 1, 3, 4)

        # Batch matrix multiplication to compute compatibilities (n_heads, batch_size, num_steps, graph_size)
        compatibility = torch.matmul(glimpse_Q, glimpse_K.transpose(-2, -1)) / math.sqrt(glimpse_Q.size(-1))
        if self.mask_inner:
            assert self.mask_logits, "Cannot mask inner without masking logits"
            compatibility[mask[None, :, :, None, :].expand_as(compatibility)] = -math.inf

        # Batch matrix multiplication to compute heads (n_heads, batch_size, num_steps, val_size)
        heads = torch.matmul(torch.softmax(compatibility, dim=-1), glimpse_V)

        # Project to get glimpse/updated context node embedding (batch_size, num_steps, embedding_dim)
        glimpse = self.project_out(
            heads.permute(1, 2, 3, 0, 4).contiguous().view(-1, num_steps, 1, self.n_heads * val_size))

        # Now projecting the glimpse is not needed since this can be absorbed into project_out
        # final_Q = self.project_glimpse(glimpse)
        final_Q = glimpse
        # Batch matrix multiplication to compute logits (batch_size, num_steps, graph_size)
        # logits = 'compatibility'
        logits = torch.matmul(final_Q, logit_K.transpose(-2, -1)).squeeze(-2) / math.sqrt(final_Q.size(-1))

        # From the logits compute the probabilities by clipping, masking and softmax
        if self.tanh_clipping > 0:
            logits = torch.tanh(logits) * self.tanh_clipping
        if self.mask_logits:
            logits[mask] = -math.inf

        return logits, glimpse.squeeze(-2)

    def _make_heads(self, v, num_steps=None):
        assert num_steps is None or v.size(1) == 1 or v.size(1) == num_steps

        return (
            v.contiguous()
            .view(v.size(0), v.size(1), v.size(2), self.n_heads, -1)
            .expand(v.size(0), v.size(1) if num_steps is None else num_steps, v.size(2), self.n_heads, -1)
            .permute(3, 0, 1, 2, 4)  # (n_heads, batch_size, num_steps, graph_size, head_dim)
        )

    @staticmethod
    def _calculate_distance_matrix(locations):
        first_location_tensor = locations.unsqueeze(dim=-2).repeat(1, 1, locations.size(1), 1)
        second_location_tensor = locations.unsqueeze(dim=-3).repeat(1, locations.size(1), 1, 1)

        return (second_location_tensor - first_location_tensor).norm(p=2, dim=-1)

    @staticmethod
    def _calculate_distance_to_depot(depot_location, node_locations):
        depot_location_extended = depot_location.unsqueeze(1).repeat(1, node_locations.size(1), 1)
        return (node_locations - depot_location_extended).norm(p=2, dim=-1)
