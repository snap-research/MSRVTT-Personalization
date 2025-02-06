
from utils.distributed.distributed_utils import initialize_distributed, cleanup_distributed
from utils.configuration.configuration_utils import parse_config

import torch
import torch.distributed as dist
import argparse
import json

torch.backends.cudnn.benchmark = True

def main():
    
    # Loads configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation_config", type=str, required=True)
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=12)
    arguments = parser.parse_args()
    evaluation_config_path = arguments.evaluation_config
    
    # Instantiates the evaluation configuration
    evaluation_config = parse_config(evaluation_config_path)

    # Performs initialization of the distributed environment
    initialize_distributed()

    evaluator_config = evaluation_config["evaluator"]
    for evaluator in evaluator_config["evaluators"]:
        evaluator["generated_video_root"] = arguments.video_folder
        evaluator["frames_count"] = arguments.num_frames
        evaluator["batch_size"] = 1 # Some evalutors only support batch size 1
        evaluator["num_workers"] = arguments.num_workers
            
    evaluator_target = evaluator_config["target"]
    evaluator = evaluator_target(evaluator_config)
    
    evaluation_results = evaluator.evaluate()
    
    if dist.get_rank() == 0:
        output_json_file = arguments.video_folder + ".json" if not arguments.video_folder.endswith("/") else arguments.video_folder[:-1] + ".json"
        with open(output_json_file, "w") as f:
            json.dump(evaluation_results, f, indent=4)
    
    # Cleans up the distributed environment
    cleanup_distributed()


if __name__ == "__main__":
    main()