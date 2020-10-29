import json
import os
import torch

from nets.attention_model import AttentionModel


def _load_model_file(load_path, model):
    """Loads the model with parameters from the file and returns optimizer state dict if it is in the file"""

    # Load the model parameters from a saved state
    load_optimizer_state_dict = None
    print('  [*] Loading model from {}'.format(load_path))

    load_data = torch.load(
        os.path.join(
            os.getcwd(),
            load_path
        ), map_location=lambda storage, loc: storage)

    if isinstance(load_data, dict):
        load_optimizer_state_dict = load_data.get('optimizer', None)
        load_model_state_dict = load_data.get('model', load_data)
    else:
        load_model_state_dict = load_data.state_dict()

    state_dict = model.state_dict()

    state_dict.update(load_model_state_dict)

    model.load_state_dict(state_dict)

    return model, load_optimizer_state_dict


def prepare_model(model_path, model_args_path, problem, decode_type='greedy'):
    with open(model_args_path, 'r') as f:
        args = json.load(f)

    model = AttentionModel(
        args['embedding_dim'],
        args['hidden_dim'],
        problem,
        cost_coefficients=args['cost_coefficients'],
        vehicle_count=args['vehicle_count'],
        n_encode_layers=args['n_encode_layers'],
        mask_inner=True,
        mask_logits=True,
        normalization=args['normalization'],
        tanh_clipping=args['tanh_clipping'],
        checkpoint_encoder=args.get('checkpoint_encoder', False),
        shrink_size=args.get('shrink_size', None)
    )

    load_data = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict({**model.state_dict(), **load_data.get('model', {})})
    model, _ = _load_model_file(model_path, model)
    model.eval()
    model.set_decode_type(decode_type)

    return model