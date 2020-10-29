from problems import CVRP, CVRPTW
from utils.model_evaluation import prepare_model
from torch.utils.data import DataLoader
from nets.bert_embedding import BertEmbedding
import torch
import configs.bert_config as configs
from itertools import permutations
from utils.bert_util import get_sample_indices
from nets.bert_transformer import BertTransformer
from nets.bert_model import BertModel
import torch.nn.functional as F
import torch.optim as optim
import math
from tensorboard_logger import Logger as TbLogger
import os
from util import torch_load_cpu


def get_vrp_route(problem, dataset, model_path, model_args_path):
    model = prepare_model(model_path, model_args_path, problem)
    data_loader = DataLoader(dataset, batch_size=configs.epoch_size)
    data = next(iter(data_loader))
    _, _, initial_route = model(data, return_pi=True)
    # depot_locs = torch.zeros((initial_route.size(0), 1), dtype=int)
    # route_start_from_depot = torch.cat((depot_locs, initial_route), dim=-1)
    return initial_route


def prepare_data(raw_data, device):
    location_tensors_list = [sample['loc'][None, :, :] for sample in raw_data]
    demand_tensors_list = [sample['demand'][None, :] for sample in raw_data]
    start_time_tensors_list = [sample['timeWindowStart'][None, :] for sample in raw_data]
    finish_time_tensors_list = [sample['timeWindowFinish'][None, :] for sample in raw_data]
    depot_location_tensor_list = [sample['depot'][None, None, :] for sample in raw_data]
    depot_start_time_tensor_list = [sample['depotStartTime'][None, :] for sample in raw_data]
    depot_finish_time_tensor_list = [sample['depotFinishTime'][None, :] for sample in raw_data]

    locations = torch.cat(location_tensors_list, dim=0)
    demands = torch.cat(demand_tensors_list, dim=0)
    start_times = torch.cat(start_time_tensors_list, dim=0)
    finish_times = torch.cat(finish_time_tensors_list, dim=0)
    depot_locations = torch.cat(depot_location_tensor_list, dim=0)
    depot_start_times = torch.cat(depot_start_time_tensor_list, dim=0)
    depot_finish_times = torch.cat(depot_finish_time_tensor_list, dim=0)
    depot_demands = -1*configs.vehicle_capacity*torch.ones(demands.size(0), 1)

    locations_with_depot = torch.cat((depot_locations, locations), dim=1).to(device)
    demands_with_depot = torch.cat((depot_demands, demands), dim=1).to(device)
    start_times_with_depot = torch.cat((depot_start_times, start_times), dim=1).to(device)
    finish_times_with_depot = torch.cat((depot_finish_times, finish_times), dim=1).to(device)

    return locations_with_depot, demands_with_depot, start_times_with_depot, finish_times_with_depot


def get_route_params(locations, demands, start_times, finish_times, route):
    route_locations = (
        locations
        .unsqueeze(1).repeat(1, route.size(1), 1, 1)
        .gather(-2, route.unsqueeze(-1).repeat(1, 1, 1, locations.size(-1)))
    )

    route_demands = (
        demands
        .unsqueeze(1).repeat(1, route.size(1), 1)
        .gather(-1, route)
    )

    route_start_times = (
        start_times
        .unsqueeze(1).repeat(1, route.size(1), 1)
        .gather(-1, route)
    )

    route_finish_times = (
        finish_times
        .unsqueeze(1).repeat(1, route.size(1), 1)
        .gather(-1, route)
    )

    return route_locations, route_demands, route_start_times, route_finish_times


def calculate_cost(dataset, route, device):
    locations, demands, start_times, finish_times = prepare_data(dataset, device)
    route_locations, route_demands, route_start_times, route_finish_times = get_route_params(locations,
                                                                                             demands,
                                                                                             start_times,
                                                                                             finish_times,
                                                                                             route)
    depot_locs = (
        locations
        .unsqueeze(1)
        .repeat(1, route.size(1), 1, 1)
        [:, :, 0:1, :]
    )

    route_eta_between_nodes = (
        torch.cat((
            configs.time_scale * (route_locations[:, :, 0:1, :] - depot_locs).norm(p=2, dim=-1),
            configs.time_scale * (route_locations[:, :, 1:, :] - route_locations[:, :, :-1, :]).norm(p=2, dim=-1),
            configs.time_scale * (route_locations[:, :, -1:, :] - depot_locs).norm(p=2, dim=-1)
        ), -1)
    )

    route_arrival_times = torch.zeros_like(route)
    for j in range(route.size(1)):
        for k in range(route.size(2)):
            if k == 0:
                route_arrival_times[:, j, k] = route_start_times[:, j, k]
            else:
                route_arrival_times[:, j, k] = route_start_times[:, j, k] * ((route[:, j, k - 1] == 0).float()) + \
                                               (torch.max(route_start_times[:, j, k - 1].float(),
                                                          route_arrival_times[:, j, k - 1].float()) +
                                                route_eta_between_nodes[:, j, k]) * ((route[:, j, k - 1] != 0).float())

    # Length is distance (L2-norm of difference) of each next location to its prev and of first and last to depot
    distance_cost = ((route_locations[:, :, 1:, :] - route_locations[:, :, :-1, :]).norm(p=2, dim=-1).sum(-1) +
                     (route_locations[:, :, 0:1, :] - depot_locs).norm(p=2, dim=-1).sum(-1) +
                     (route_locations[:, :, -1:, :] - depot_locs).norm(p=2, dim=-1).sum(-1))

    zero_tensor = torch.zeros_like(route_arrival_times)
    early_arrival_cost = torch.max((route_start_times - route_arrival_times).float(), zero_tensor.float()).sum(dim=-1)
    delay_time_cost = torch.max((route_arrival_times - route_finish_times).float(), zero_tensor.float()).sum(dim=-1)
    total_cost = configs.distance_cost_coefficient*distance_cost + \
                 configs.early_cost_coefficient*early_arrival_cost + \
                 configs.delay_cost_coefficient*delay_time_cost

    return total_cost, distance_cost, delay_time_cost


def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def log_values(tb_logger, cost, distance_cost, delay_cost, cost_improvement, distance_cost_improvement,
               delay_cost_improvement, loss, batch_id):
    tb_logger.log_value('cost', cost, batch_id)
    tb_logger.log_value('distance_cost', distance_cost, batch_id)
    tb_logger.log_value('delay_cost', delay_cost, batch_id)
    tb_logger.log_value('cost improvement', cost_improvement, batch_id)
    tb_logger.log_value('distance_cost improvement', distance_cost_improvement, batch_id)
    tb_logger.log_value('delay_cost improvement', delay_cost_improvement, batch_id)
    tb_logger.log_value('loss', loss, batch_id)


def train(model, optimizer, num_epoch, tb_logger, device):
    batch_id = 0
    for i in range(num_epoch):
        print('epoch {} started.'.format(i))
        for j in range(configs.epoch_size//configs.batch_size):
            cost, distance_cost, delay_cost, cost_improvement, distance_cost_improvement, delay_cost_improvement, \
            loss = train_batch(model, optimizer, device)
            print('cost: {}, distance_cost: {}, delay_cost: {}, loss: {}'.format(cost_improvement,
                                                                                 distance_cost_improvement,
                                                                                 delay_cost_improvement, loss))
            batch_id += 1

            if batch_id % configs.log_step == 0:
                log_values(tb_logger, cost, distance_cost, delay_cost, cost_improvement, distance_cost_improvement,
                           delay_cost_improvement, loss, batch_id)
    # selected_route = (
    #     possible_routes
    #     .gather(1, selected_indices.unsqueeze(1).unsqueeze(-1).repeat(1, 1, possible_routes.size(-1)))
    #     .squeeze(1)
    # )


def save_model(model, path):
    torch.save(
        {'model': model.state_dict()}, path
    )


def get_dataset(data_problem = CVRPTW, model_problem = CVRP):
    training_dataset = data_problem.make_dataset(size=configs.graph_size, num_samples=configs.batch_size)
    training_dataset_route = get_vrp_route(problem=model_problem, dataset=training_dataset,
                                           model_path=configs.model_path, model_args_path=configs.model_args_path)

    return training_dataset, training_dataset_route


def train_batch(model, optimizer, device):
    training_dataset, training_dataset_route = get_dataset()
    initial_cost, initial_distance_cost, initial_delay_cost = calculate_cost(training_dataset.data,
                                                                             training_dataset_route.unsqueeze(1).to(device),
                                                                             device)

    possible_routes, likelihood, capacity_mask = model(training_dataset, training_dataset_route)
    likelihood = likelihood.clamp(min=configs.min_likelihood)
    cost, distance_cost, delay_cost = calculate_cost(training_dataset.data, possible_routes, device)
    selected_indices = cost.argmin(-1)

    selected_indices_cost = cost.gather(1, selected_indices.unsqueeze(1)).squeeze(1)
    selected_distance_cost = distance_cost.gather(1, selected_indices.unsqueeze(1)).squeeze(1)
    selected_delay_cost = delay_cost.gather(1, selected_indices.unsqueeze(1)).squeeze(1)

    cost_improvement = (initial_cost.squeeze(1) - selected_indices_cost).mean()
    distance_cost_improvement = (initial_distance_cost.squeeze(1) - selected_distance_cost).mean()
    delay_cost_improvement = (initial_delay_cost.squeeze(1) - selected_delay_cost).mean()

    loss = ((cost * likelihood * capacity_mask).sum(1)/(capacity_mask.sum(1))).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return selected_indices_cost.mean(), selected_distance_cost.mean(), selected_delay_cost.mean(), \
        cost_improvement, distance_cost_improvement, delay_cost_improvement, loss


def evaluate(model, model_path):
    load_data = torch_load_cpu(model_path)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    evaluation_dataset, evaluation_dataset_route = get_dataset()
    possible_routes, likelihood, capacity_mask = model(evaluation_dataset, evaluation_dataset_route)
    print(possible_routes.size())


if __name__ == '__main__':
    device = torch.device("cuda:0" if (configs.use_cuda and torch.cuda.is_available()) else "cpu")
    run_name = 'test'
    eval = True
    bert_model = BertModel(configs.embedding_dim, configs.sample_size, configs.vehicle_capacity, device).to(device)
    model_path = os.path.join(configs.model_output_dir, run_name, 'model.pt')
    if eval:
        evaluate(bert_model, model_path)
    else:
        optimizer = optim.Adam([{'params': bert_model.parameters(), 'lr': configs.lr}])

        tb_logger = TbLogger(os.path.join(configs.log_dir, "{}_{}".format('bert', configs.graph_size), run_name))
        train(bert_model, optimizer, configs.num_epochs, tb_logger, device)
        if not os.path.exists(model_path):
            os.makedirs(os.path.dirname(model_path))

        save_model(bert_model, model_path)
