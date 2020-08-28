import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateCVRP(NamedTuple):
    # Fixed input
    coords: torch.Tensor  # Depot + loc
    demand: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the coords and demands tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    used_capacity: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    lengths: torch.Tensor
    cur_coord: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    VEHICLE_CAPACITY = 1.0  # Hardcoded

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.demand.size(-1))

    @property
    def dist(self):
        return (self.coords[:, :, None, :] - self.coords[:, None, :, :]).norm(p=2, dim=-1)

    def __getitem__(self, key):
        if torch.is_tensor(key) or isinstance(key, slice):  # If tensor, idx all tensors by this tensor:
            return self._replace(
                ids=self.ids[key],
                prev_a=self.prev_a[key],
                used_capacity=self.used_capacity[key],
                visited_=self.visited_[key],
                lengths=self.lengths[key],
                cur_coord=self.cur_coord[key],
            )
        return super(StateCVRP, self).__getitem__(key)

    # Warning: cannot override len of NamedTuple, len should be number of fields, not batch size
    # def __len__(self):
    #     return len(self.used_capacity)

    @staticmethod
    def initialize(input, vehicle_count, visited_dtype=torch.uint8):

        depot = input['depot']
        loc = input['loc']
        demand = input['demand']

        batch_size, n_loc, _ = loc.size()
        return StateCVRP(
            coords=torch.cat((depot[:, None, :], loc), -2),
            demand=demand,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            prev_a=torch.zeros(batch_size, 1, vehicle_count, dtype=torch.long, device=loc.device),
            used_capacity=demand.new_zeros(batch_size, 1, vehicle_count),
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                # Keep visited_ with depot so we can scatter efficiently
                torch.zeros(
                    batch_size, 1, (n_loc + 1)*vehicle_count,
                    dtype=torch.uint8, device=loc.device
                )
                if visited_dtype == torch.uint8
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            lengths=torch.zeros(batch_size, 1, vehicle_count, device=loc.device),
            cur_coord=(input['depot'][:, None, None, :]).repeat([1, 1, vehicle_count, 1]),  # Add step dimension
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.lengths + ((self.coords[self.ids, 0, None, :] -
                               self.cur_coordself.cur_coord[self.ids, 0, :, :]).norm(p=2, dim=-1)).sum(dim=-1)

    def update(self, selected, vehicle_count):

        assert self.i.size(0) == 1, "Can only update if state represents single step"

        # Update the state
        vehicle_index = selected % vehicle_count
        selected_node = selected // vehicle_count
        selected = selected_node[:, None]  # Add dimension for step
        prev_a = selected
        n_loc = self.demand.size(-1)  # Excludes depot

        # Add the length
        cur_coord = self.coords[self.ids, selected]
        # cur_coord = self.coords.gather(
        #     1,
        #     selected[:, None].expand(selected.size(0), 1, self.coords.size(-1))
        # )[:, 0, :]
        lengths = self.lengths[self.ids, 0, vehicle_index] + \
            (cur_coord - self.cur_coord[self.ids, 0, vehicle_index]).norm(p=2, dim=-1)  # (batch_dim, 1)

        # Not selected_demand is demand of first node (by clamp) so incorrect for nodes that visit depot!
        #selected_demand = self.demand.gather(-1, torch.clamp(prev_a - 1, 0, n_loc - 1))
        selected_demand = self.demand[self.ids, torch.clamp(prev_a - 1, 0, n_loc - 1)]

        # Increase capacity if depot is not visited, otherwise set to 0
        #used_capacity = torch.where(selected == 0, 0, self.used_capacity + selected_demand)
        used_capacity = (self.used_capacity[self.ids, 0, vehicle_index] + selected_demand) * (prev_a != 0).float()

        if self.visited_.dtype == torch.uint8:
            # Note: here we do not subtract one as we have to scatter so the first column allows scattering depot
            # Add one dimension since we write a single value
            visited_locations = vehicle_count * prev_a[:, :, None]
            for i in range(1, vehicle_count):
                visited_locations = torch.cat((visited_locations, vehicle_count * prev_a[:, :, None] + i), -1)

            visited_ = self.visited_.scatter(-1, visited_locations, 1)
        else:
            # This works, will not set anything if prev_a -1 == -1 (depot)
            visited_ = self.visited_
            for i in range(vehicle_count):
                visited_ = mask_long_scatter(visited_, (vehicle_count * (prev_a - 1) + i).clamp(min=-1))

        prev_a_tmp = self.prev_a
        prev_a_tmp[self.ids, 0, vehicle_index] = prev_a

        used_capacity_tmp = self.used_capacity
        used_capacity_tmp[self.ids, 0, vehicle_index] = used_capacity

        lengths_tmp = self.lengths
        lengths_tmp[self.ids, 0, vehicle_index] = lengths

        cur_coord_tmp = self.cur_coord
        cur_coord_tmp[self.ids, 0, vehicle_index] = cur_coord

        return self._replace(
            prev_a=prev_a_tmp, used_capacity=used_capacity_tmp, visited_=visited_,
            lengths=lengths_tmp, cur_coord=cur_coord_tmp, i=self.i + 1
        )

    def all_finished(self):
        return self.i.item() >= self.demand.size(-1) and self.visited.all()

    def get_finished(self):
        return self.visited.sum(-1) == self.visited.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self, vehicle_count):
        """
        Gets a (batch_size, n_loc + 1) mask with the feasible actions (0 = depot), depends on already visited and
        remaining capacity. 0 = feasible, 1 = infeasible
        Forbids to visit depot twice in a row, unless all nodes have been visited
        :return:
        """

        if self.visited_.dtype == torch.uint8:
            visited_loc = self.visited_[:, :, vehicle_count:]
        else:
            visited_loc = mask_long2bool(self.visited_, n=self.demand.size(-1))

        demands = (self.demand.view(-1, 1).repeat(1, vehicle_count).view(self.ids.shape[0], -1))[self.ids, :]
        used_cap = (
            self.used_capacity
            .view(self.used_capacity.shape[0], 1, vehicle_count)
            .repeat(1, 1, demands.shape[-1] // vehicle_count)
        )
        # For demand steps_dim is inserted by indexing with id, for used_capacity insert node dim for broadcasting
        exceeds_cap = (demands + used_cap > self.VEHICLE_CAPACITY)
        # Nodes that cannot be visited are already visited or too much demand to be served now
        mask_loc = visited_loc.to(exceeds_cap.dtype) | exceeds_cap

        # Cannot visit the depot if just visited and still unserved nodes
        mask_depot = (self.prev_a == 0) & ((mask_loc == 0).int().sum(-1) > 0)[:, :, None]
        return torch.cat((mask_depot, mask_loc), -1)

    def construct_solutions(self, actions):
        return actions
