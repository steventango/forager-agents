import glob
import json
import os

import numpy as np
from RlEvaluation.hypers import HyperSelectionResult


def update_best_config(alg: str, report: HyperSelectionResult, file: str):
    dir_path = os.path.dirname(file)
    sweep_path = glob.glob(f"{dir_path}/**/{alg}.json", recursive=True)[0]
    path = sweep_path.replace("-sweep", "")
    with open(sweep_path, "r") as f:
        sweep_config = json.load(f)
    with open(path, "r") as f:
        config = json.load(f)

    for config_param, best_config in zip(report.config_params, report.best_configuration):
        parts = config_param.split(".")
        curr = config["metaParameters"]
        curr_sweep = sweep_config["metaParameters"]
        for part in parts[:-1]:
            curr = curr[part]
            curr_sweep = curr_sweep[part]
        if not isinstance(curr_sweep[parts[-1]], list):
            continue
        if isinstance(best_config, np.integer):
            best_config = int(best_config)
        elif isinstance(best_config, np.floating):
            best_config = float(best_config)
        curr[parts[-1]] = best_config

    with open(path, "w") as f:
        json.dump(config, f, indent=4)
