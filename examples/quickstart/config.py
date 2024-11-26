#================================================================
# Evaluator
#================================================================

import cognify

metric = cognify.metric.f1_score_str

@cognify.register_evaluator
def evaluate_answer(answer, label):
    return metric(answer, label)

#================================================================
# Data Loader
#================================================================

import json

@cognify.register_data_loader
def load_data_minor():
    with open("data._json", "r") as f:
        data = json.load(f)
          
    # format to (input, output) pairs
    new_data = []
    for d in data:
        input = {
            'question': d["question"], 
            'documents': d["docs"]
        }
        output = {
            'label': d["label"],
        }
        new_data.append((input, output))
    return new_data[:5], None, new_data[5:]

#================================================================
# Optimizer Set Up
#================================================================

from cognify.hub.search import default

search_settings = default.create_search(
    search_type='medium',
    n_trials=30,
)
