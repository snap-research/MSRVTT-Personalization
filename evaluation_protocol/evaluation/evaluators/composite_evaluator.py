from typing import Dict

import torch
import torch.nn as nn
import torch.distributed as dist
from utils.distributed.distributed_utils import print_r0

class CompositeEvaluator():
    """
    Class representing a set of evaluators executed sequentially
    """
    def __init__(self, evaluator_config: Dict):
        evaluators_configs = evaluator_config["evaluators"]
        self.evaluators = []
        for current_evaluator_config in evaluators_configs:
            current_target = current_evaluator_config["target"]
            current_evaluator = current_target(current_evaluator_config)
            self.evaluators.append(current_evaluator)

    def evaluate(self):
        """
        Performs evaluation
        """
        all_results = {}
        
        # Executes each evaluator and merges the results
        for current_evaluator in self.evaluators:
            current_results = current_evaluator.evaluate()
            if current_results is not None: # Processes != rank0 may return None
                print_r0("  - Partial evaluator results in CompositeEvaluator: {}".format(current_results))
                all_results.update(current_results)

        return all_results
