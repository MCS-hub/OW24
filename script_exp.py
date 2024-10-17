import argparse
from src.experiments import (
    conv_comp_icnn_jko_dc_mix_gauss_targ_experiment,
    distance_to_set_prior_experiment,
    conv_comp_em_mix_gauss_targ_experiment,
)
import yaml


def extract_method(config, args):
    method = args.method
    discretization = args.discretization
    methods_supported = list(config["method"].keys())
    assert (
        method in methods_supported
    ), f"method {method} not supported, consider supported list : {methods_supported}"
    dict_method = config.pop("method")[method]
    config["experiment_method"] = method
    config["discretization"] = discretization
    return {**config, **dict_method}


def extract_exp_ns(config, args):
    exp_ns = args.exp_ns
    exps_count = config["exps_count"]
    if exp_ns == -1:
        exp_ns = list(range(exps_count))
    else:
        if isinstance(exp_ns, int) and exp_ns > 0:
            exp_ns = list(range(exp_ns))
        else:
            raise ValueError("invalid exp_ns")
    config["exp_numbers"] = exp_ns
    return config


def extract_verbose(config, args):
    config["verbose"] = args.verbose
    return config


def extract_device(config, args):
    config["device"] = args.device
    return config


parser = argparse.ArgumentParser(description="Runs our experiments")
parser.add_argument("experiment", help="experiment name")
parser.add_argument(
    "--method",
    help="method solving the task (if needed)",
    type=str,
    default="ICNN_jko_dc",
)
# parser.add_argument("discretization", help="discretization name, either fb or semi_fb")
parser.add_argument(
    "--discretization",
    type=str,
    choices=["semi_fb", "fb"],
    help="Type of discretization",
)
parser.add_argument(
    "--exp_ns",
    type=int,
    help="number of experiments to peform (if needed)",
    default=-1,
)
parser.add_argument(
    "--verbose", dest="verbose", action="store_const", const=True, default=True
)
parser.add_argument(
    "--device",
    action="store",
    help="device (for NN training)",
    type=str,
    default="cuda:0",
)

experiment_map = {
    "relaxed_vmF": {
        "config_path": "./configs/relaxed_vmF.yml",
        "preprocess": [extract_method, extract_exp_ns, extract_verbose, extract_device],
        "function": {"ICNN_jko_dc": distance_to_set_prior_experiment},
    },
    "conv_comp_dim_2": {
        "config_path": "./configs/convergence_comparison_dim_2.yml",
        "preprocess": [extract_method, extract_exp_ns, extract_verbose, extract_device],
        "function": {
            "ICNN_jko_dc": conv_comp_icnn_jko_dc_mix_gauss_targ_experiment,
            "EM_sim_10000": conv_comp_em_mix_gauss_targ_experiment,
        },
    },
}


# Experiments
experiment = "conv_comp_dim_2"
exp_ns = 5
verbose = True
device = "cuda:0"

# EM
method = "EM_sim_10000"
args = argparse.Namespace(
    experiment=experiment, method=method, exp_ns=exp_ns, verbose=verbose, device=device
)
exp_name = args.experiment
assert (
    exp_name in experiment_map
), f"Experiment '{exp_name}' not defined, consider one in the list: '{list(experiment_map.keys())}'"
tech_config = experiment_map[exp_name]
config_path = tech_config["config_path"]
with open(config_path, "r") as fp:
    config = yaml.full_load(fp)
for func in tech_config["preprocess"]:
    config = func(config, args)
function = tech_config["function"]

function[method](config)


# ICNN_jko_dc
method = "ICNN_jko_dc"
args = argparse.Namespace(
    experiment=experiment, method=method, exp_ns=exp_ns, verbose=verbose, device=device
)

exp_name = args.experiment
assert (
    exp_name in experiment_map
), f"Experiment '{exp_name}' not defined, consider one in the list: '{list(experiment_map.keys())}'"
tech_config = experiment_map[exp_name]
config_path = tech_config["config_path"]
with open(config_path, "r") as fp:
    config = yaml.full_load(fp)
for func in tech_config["preprocess"]:
    config = func(config, args)
function = tech_config["function"]
# FB-Euler
config["discretization"] = "fb"
function[method](config)
# semi FB-Euler
config["discretization"] = "semi_fb"
function[method](config)
