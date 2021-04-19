import json
import os

def save_experiment(config: dict, metrics: dict, name: str, save_dir="../experiments"):
    os.makedirs(save_dir, exist_ok=True)
    name = name.replace(save_dir,"")
    base_dir = os.path.join(save_dir, name)
    os.makedirs(base_dir, exist_ok=True)
    with open(os.path.join(base_dir, "parameters.json"), "w") as save_file:
        json.dump(config, save_file)
    
    # for metric_type in ["train", "validation", "test"]:
    #     if metric_type in metrics:
    for metric_type in metrics.keys():
        with open(os.path.join(base_dir, f"{metric_type}.json"), "w") as save_file:
            json.dump(metrics[metric_type], save_file, indent=4)