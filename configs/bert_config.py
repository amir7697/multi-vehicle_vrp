data_path = 'data/vrp/vrp20_test_seed1111.pkl'
model_path = 'evaluation/trained_models/cvrp_20/location_embedding/epoch-99.pt'
model_args_path = 'evaluation/trained_models/cvrp_20/location_embedding/args.json'
embedding_dim = 768
min_likelihood = -1000
model_output_dir = 'outputs/bert_20'

sample_size = 5
vehicle_capacity = 30
lr = 0.0001

num_epochs = 1

graph_size = 20
epoch_size = 2
batch_size = 2

time_horizon = 1000
vehicle_count = 1
time_scale = 100

distance_cost_coefficient = 100
early_cost_coefficient = 0
delay_cost_coefficient = 0.5
max_grad_norm = 1

log_step = 10
log_dir = 'logs'

use_cuda = True