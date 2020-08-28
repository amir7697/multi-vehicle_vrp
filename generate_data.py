import argparse
import os
import numpy as np
from utils.data_utils import check_extension, save_dataset


def generate_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_vrptw_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.
    }

    SERVICE_TIME = 0
    TIME_HORIZON = 1000
    TIME_SCALE = 100

    depot = np.random.uniform(size=(dataset_size, 2))  # Depot location
    loc = np.random.uniform(size=(dataset_size, vrp_size, 2))  # Node locations
    demand = np.random.randint(1, 10, size=(dataset_size, vrp_size))  # Demand, uniform integer 1 ... 9
    capacity = np.full(dataset_size, CAPACITIES[vrp_size])

    depot_start_time = np.zeros(dataset_size)
    depot_finish_time = TIME_HORIZON*np.ones(dataset_size)
    service_time = SERVICE_TIME*np.ones((dataset_size, vrp_size))

    time_window_start_time = np.zeros((dataset_size, vrp_size))
    time_window_finish_time = np.zeros((dataset_size, vrp_size))

    for i in range(dataset_size):
        customer_eta_to_depot = TIME_SCALE*np.linalg.norm(loc[i] - depot[i], axis=1)
        customer_start_time_horizon = depot_start_time[i] + customer_eta_to_depot + 1
        customer_finish_time_horizon = depot_finish_time[i] - customer_eta_to_depot

        epsilon = np.clip(abs(np.random.randn(vrp_size)), a_min=0.01, a_max=np.inf)
        time_window_start_time[i] = [np.random.randint(customer_start_time_horizon[i], customer_finish_time_horizon[i])
                                     for i in range(vrp_size)]
        time_window_finish_time[i] = np.minimum(time_window_start_time[i] + 300*epsilon, customer_finish_time_horizon)

    return list(zip(
        depot.tolist(),  # Depot location
        loc.tolist(),  # Node locations
        demand.tolist(),  # Demand, uniform integer 1 ... 9
        capacity.tolist(),  # Capacity, same for whole dataset
        depot_start_time.tolist(),
        depot_finish_time.tolist(),
        service_time.tolist(),
        time_window_start_time.tolist(),
        time_window_finish_time.tolist()
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--name", type=str, required=True, help="Name to identify dataset")
    parser.add_argument("--problem", type=str, default='all', help="Problem, 'vrp', 'vrptw' or 'all' to generate all")
    parser.add_argument('--data_distribution', type=str, default='all',
                        help="Distributions to generate for problem, default 'all'.")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--graph_sizes', type=int, nargs='+', default=[20, 50, 100],
                        help="Sizes of problem instances (default 20, 50, 100)")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    distributions_per_problem = {
        'vrp': [None],
        'vrptw': [None]
    }
    if opts.problem == 'all':
        problems = distributions_per_problem
    else:
        problems = {
            opts.problem:
                distributions_per_problem[opts.problem]
                if opts.data_distribution == 'all'
                else [opts.data_distribution]
        }

    for problem, distributions in problems.items():
        for distribution in distributions or [None]:
            for graph_size in opts.graph_sizes:

                datadir = os.path.join(opts.data_dir, problem)
                os.makedirs(datadir, exist_ok=True)

                if opts.filename is None:
                    filename = os.path.join(datadir, "{}{}{}_{}_seed{}.pkl".format(
                        problem,
                        "_{}".format(distribution) if distribution is not None else "",
                        graph_size, opts.name, opts.seed))
                else:
                    filename = check_extension(opts.filename)

                assert opts.f or not os.path.isfile(check_extension(filename)), \
                    "File already exists! Try running with -f option to overwrite."

                np.random.seed(opts.seed)
                if problem == 'vrp':
                    dataset = generate_vrp_data(opts.dataset_size, graph_size)
                elif problem == 'vrptw':
                    dataset = generate_vrptw_data(opts.dataset_size, graph_size)
                else:
                    assert False, "Unknown problem: {}".format(problem)

                save_dataset(dataset, filename)
