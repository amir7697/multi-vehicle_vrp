from torch.utils.data import Dataset
import torch
import os
import pickle

from problems.vrp.state_vrp import StateCVRP


class CVRP(object):
    NAME = 'cvrp'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi, cost_coefficients, vehicle_count):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        corrected_pi = pi // vehicle_count
        sorted_pi = corrected_pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRP.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )[:, None, :].repeat(1, vehicle_count, 1)

        route = torch.zeros(pi.size(0), vehicle_count, pi.size(1), dtype=int, device=demand_with_depot.device)
        for i, way in enumerate(pi):
            for j in range(vehicle_count):
                temp = way[way % vehicle_count == j]
                route[i, j, :temp.size(0)] = temp // vehicle_count

        d = demand_with_depot.gather(-1, route)

        for j in range(vehicle_count):
            used_cap = torch.zeros_like(dataset['demand'][:, 0])
            for i in range(route.size(2)):
                used_cap += d[:, j, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
                # Cannot use less than 0
                used_cap[used_cap < 0] = 0
                assert (used_cap <= CVRP.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = (
            torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
            [:, None, :, :].repeat(1, vehicle_count, 1, 1)
        )

        locations = loc_with_depot.gather(2, route[..., None].expand(*route.size(), loc_with_depot.size(-1)))

        distance_cost = ((locations[:, :, 1:] - locations[:, :, :-1]).norm(p=2, dim=-1).sum(-1)\
                         + (locations[:, :, 0:1] - dataset['depot'][:, None, None, :]).norm(p=2, dim=-1).sum(-1)\
                         + ((locations[:, :, -1:] - dataset['depot'][:, None, None, :]).norm(p=2, dim=-1)).sum(-1)
                         ).sum(dim=-1)

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        return cost_coefficients['distance']*distance_cost, None, cost_coefficients['distance']*distance_cost, 0, 0

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRP.initialize(*args, **kwargs)


def make_instance(args):
    depot, loc, demand, capacity, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size
    }


class VRPDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.data = [make_instance(args) for args in data[offset:offset + num_samples]]

        else:

            # From VRP with RL paper https://arxiv.org/abs/1802.04240
            CAPACITIES = {
                10: 20.,
                20: 30.,
                50: 40.,
                100: 50.
            }

            locs = torch.FloatTensor(num_samples, size, 2).uniform_(0, 1)
            demand = (torch.FloatTensor(num_samples, size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size]
            depot = torch.FloatTensor(num_samples, 2).uniform_(0, 1)

            self.data = [
                {
                    'loc': locs[i],
                    # Uniform 1 - 9, scaled by capacities
                    'demand': demand[i],
                    'depot': depot[i]
                }
                for i in range(num_samples)
            ]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
