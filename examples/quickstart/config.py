#================================================================
# Evaluator
#================================================================

import cognify
from cognify.hub.evaluators import f1_score_str

@cognify.register_evaluator
def evaluate_answer(answer, label):
    return f1_score_str(answer, label)

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

# from cognify.hub.search import default

# search_settings = default.create_search(
#     search_type='light',
#     n_trials=5,
#     evaluator_batch_size=10,
# )


from cognify.optimizer.core import driver, flow
from cognify.hub.cogs import reasoning
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import ZeroShotCoT
from cognify.optimizer.control_param import ControlParameter, SelectedObjectives

reasoning_param = reasoning.LMReasoning([NoChange(), ZeroShotCoT()])
outer_opt_config = flow.OptConfig(
    n_trials=16,
    throughput=4,
    use_SH_allocation=True,
    # use_HB_allocation=True,
    initial_step_budget=4,
)
params = [reasoning_param]
outer_loop_config = driver.LayerConfig(
    layer_name="outer",
    universal_params=params,
    opt_config=outer_opt_config,
)


few_shot_params = LMFewShot(2)
inner_opt_config = flow.OptConfig(
    n_trials=0,
    patience=flow.PatienceConfig(0.02,0.05,0.05,2)
)
inner_loop_config = driver.LayerConfig(
    layer_name="inner",
    universal_params=[reasoning_param],
    opt_config=inner_opt_config,
)

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_layer_configs=[outer_loop_config, inner_loop_config],
    objectives=SelectedObjectives(True, True, True),
    opt_history_log_dir="test_SH",
    evaluator_batch_size=10,
    quality_constraint=0.99,
)