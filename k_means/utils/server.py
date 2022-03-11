"""Module that helps organize the core.server module."""
import json
from k_means.core.algorithm import main_k_means_algorithm
from k_means.utils.validation import validate_message_vs_schema


def run_simulation_util(message):
    note, result_validation = validate_message_vs_schema(message,
                                                         'runSimulationSchema.json')
    if not result_validation:
        return json.dumps(note)
    # else, the message has been validated and we can continue
    result_algo = main_k_means_algorithm(message)
    return json.dumps(result_algo)
