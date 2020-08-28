import torch
import numpy as np
import os
import pickle

from torch.utils.data import Dataset
from problems.vrptw.state_vrptw import StateCVRPTW


TIME_SCALE = 100


class CVRPTW(object):

    NAME = 'cvrptw'  # Capacitated Vehicle Routing Problem

    VEHICLE_CAPACITY = 1.0  # (w.l.o.g. vehicle capacity is 1, demands should be scaled)

    @staticmethod
    def get_costs(dataset, pi, cost_coefficients, vehicle_count=1):
        batch_size, graph_size = dataset['demand'].size()
        # Check that tours are valid, i.e. contain 0 to n -1
        corrected_pi = pi//vehicle_count
        sorted_pi = corrected_pi.data.sort(1)[0]

        # Sorting it should give all zeros at front and then 1...n
        assert (
            torch.arange(1, graph_size + 1, out=pi.data.new()).view(1, -1).expand(batch_size, graph_size) ==
            sorted_pi[:, -graph_size:]
        ).all() and (sorted_pi[:, :-graph_size] == 0).all(), "Invalid tour"

        # Visiting depot resets capacity so we add demand = -capacity (we make sure it does not become negative)
        demand_with_depot = torch.cat(
            (
                torch.full_like(dataset['demand'][:, :1], -CVRPTW.VEHICLE_CAPACITY),
                dataset['demand']
            ),
            1
        )[:, None, :].repeat(1, vehicle_count, 1)
        route = torch.zeros(pi.size(0), vehicle_count, pi.size(1), dtype=int)
        for i, way in enumerate(pi):
            cursur = vehicle_count*[0]
            for j, location in enumerate(way):
                route[i, location % vehicle_count, cursur[location % vehicle_count]] = location//vehicle_count
                cursur[location % vehicle_count] += 1

        d = demand_with_depot.gather(-1, route)

        used_cap = torch.zeros_like(dataset['demand'][:, 0])
        for j in range(vehicle_count):
            for i in range(route.size(2)):
                used_cap += d[:, j, i]  # This will reset/make capacity negative if i == 0, e.g. depot visited
                # Cannot use less than 0
                used_cap[used_cap < 0] = 0
                assert (used_cap <= CVRPTW.VEHICLE_CAPACITY + 1e-5).all(), "Used more than capacity"

        # Gather dataset in order of tour
        loc_with_depot = (
            torch.cat((dataset['depot'][:, None, :], dataset['loc']), 1)
            [:, None, :, :].repeat(1, vehicle_count, 1, 1)
        )

        start_time_with_depot = (
            torch.cat((dataset['depotStartTime'], dataset['timeWindowStart']), 1)
            [:, None, :].repeat(1, vehicle_count, 1)
        )
        finish_time_with_depot = (
            torch.cat((dataset['depotFinishTime'], dataset['timeWindowFinish']), 1)
            [:, None, :].repeat(1, vehicle_count, 1)
        )
        service_time_with_depot = (
            torch.cat((torch.zeros((batch_size, 1), device=loc_with_depot.device), dataset['serviceTime']), 1)
            [:, None, :].repeat(1, vehicle_count, 1)
        )

        locations = loc_with_depot.gather(2, route[..., None].expand(*route.size(), loc_with_depot.size(-1)))
        service_times = service_time_with_depot.gather(2, route)
        start_times = start_time_with_depot.gather(2, route)
        finish_times = finish_time_with_depot.gather(2, route)

        eta_matrix = torch.cat((
                TIME_SCALE * (locations[:, :, 0:1] - dataset['depot'][:, None, None, :]).norm(p=2, dim=-1),
                TIME_SCALE * (locations[:, :, 1:] - locations[:, :, :-1]).norm(p=2, dim=-1)
            ), -1)

        arrival_times = torch.zeros_like(route)

        for i in range(route.size(0)):
            for j in range(route.size(1)):
                arrival_times[i, j, 0] = start_times[i, j, 0]
                for k in range(1, route.size(2)):
                    if route[i, j, k-1] == 0:
                        arrival_times[i, j, k] = start_times[i, j, k]
                    else:
                        arrival_times[i, j, k] = max(start_times[i, j, k-1], arrival_times[i, j, k-1]) + \
                                                     eta_matrix[i, j, k]

        # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
        distance_cost = ((locations[:, :, 1:] - locations[:, :, :-1]).norm(p=2, dim=-1).sum(-1)\
                         + (locations[:, :, 0:1] - dataset['depot'][:, None, None, :]).norm(p=2, dim=-1).sum(-1)\
                         + ((locations[:, :, -1:] - dataset['depot'][:, None, None, :]).norm(p=2, dim=-1)).sum(-1)
                         ).sum(dim=-1)

        service_time_cost = service_times.sum(dim=[-1, -2])
        early_arrival_time = (start_times - arrival_times)*((start_times - arrival_times) > 0).int()
        early_arrival_cost = early_arrival_time.sum(dim=[-1, -2])
        delay_time = (arrival_times - finish_times)*((arrival_times - finish_times) > 0).int()
        delay_time_cost = delay_time.sum(dim=[-1, -2])

        total_cost = cost_coefficients['distance']*distance_cost + cost_coefficients['service']*service_time_cost + \
            cost_coefficients['early']*early_arrival_cost + cost_coefficients['delay']*delay_time_cost
        return total_cost, None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return VRPTWDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateCVRPTW.initialize(*args, **kwargs)


def make_instance(args):
    depot, loc, demand, capacity, depot_start_time, depot_finish_time, service_time, time_window_start, \
     time_window_finish, *args = args
    grid_size = 1
    if len(args) > 0:
        depot_types, customer_types, grid_size = args
    return {
        'loc': torch.tensor(loc, dtype=torch.float) / grid_size,
        'demand': torch.tensor(demand, dtype=torch.float) / capacity,
        'depot': torch.tensor(depot, dtype=torch.float) / grid_size,
        'depotStartTime': torch.tensor([depot_start_time], dtype=torch.float),
        'depotFinishTime': torch.tensor([depot_finish_time], dtype=torch.float),
        'serviceTime': torch.tensor(service_time, dtype=torch.float),
        'timeWindowStart': torch.tensor(time_window_start, dtype=torch.float),
        'timeWindowFinish': torch.tensor(time_window_finish, dtype=torch.float)
    }


class VRPTWDataset(Dataset):

    def __init__(self, filename=None, size=50, num_samples=1000000, offset=0, distribution=None):
        super(VRPTWDataset, self).__init__()

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

            # todo: add service time
            SERVICE_TIME = 0
            TIME_HORIZON = 1000

            self.data = [
                {
                    # 'loc': (torch.randint(0, 100, (size, 2)).float())/100,
                    'loc': torch.rand((size, 2)),
                    'demand': (torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / CAPACITIES[size],
                    'depot': torch.rand((2, )),
                    'depotStartTime': torch.zeros(1),
                    'depotFinishTime': TIME_HORIZON * torch.ones(1),
                    'serviceTime': torch.cat((torch.zeros(1), SERVICE_TIME * torch.ones(size)), -1)
                }
                for i in range(num_samples)
            ]

            for sample in self.data:
                customer_eta_to_depot = self.calculate_eta(sample['loc'], sample['depot'].view(1, 2).expand(size, 2))
                customer_horizon_start_time = (sample['depotStartTime'] + customer_eta_to_depot + 1).numpy()
                customer_horizon_finish_time = sample['depotFinishTime'].numpy()[0] - customer_horizon_start_time

                noise = torch.abs(torch.randn(size))
                duration_threshold = torch.FloatTensor([0.01])
                epsilon = torch.max(noise, duration_threshold.expand_as(noise))

                sample['timeWindowStart'] = torch.tensor(
                    [np.random.uniform(customer_horizon_start_time[i], customer_horizon_finish_time[i]) for
                     i in range(size)])

                sample['timeWindowFinish'] = torch.min((sample['timeWindowStart'] + 300*epsilon),
                                                       torch.tensor(customer_horizon_finish_time))

        self.size = len(self.data)

    @staticmethod
    def calculate_eta(first_locs, second_locs):
        return TIME_SCALE*(first_locs - second_locs).norm(p=2, dim=-1)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
