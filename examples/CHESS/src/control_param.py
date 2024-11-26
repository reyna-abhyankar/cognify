quality_constraint: float = 0.98
train_down_sample: int = 50
val_down_sample: int = 25
opt_history_log_dir: str = 'cognify_opt_results'
evaluator_parallel: int = 20

from cognify.optimizer.core import driver, flow

from cognify.hub.cogs import reasoning, model_selection
from cognify.hub.cogs import ensemble
from cognify.hub.cogs.common import NoChange
from cognify.hub.cogs.fewshot import LMFewShot
from cognify.hub.cogs.reasoning import ZeroShotCoT, PlanBefore

"""
Be aware that for this workload a single trial running 100 input data requres $0.5~$1.0, please be cautious about the cost.

In this example we limit the ensemble parameter to be applied only on three modules with some prior knowledge to save cost.
"""

# ================= Reasoning Options =================
reasoning_param = reasoning.LMReasoning(
    "reasoning", [NoChange(), ZeroShotCoT(), PlanBefore()] 
)
# ================= Few Shot Options =================
few_shot_params = LMFewShot("few_shot", 4)

# ================= Inner Loop Config =================
inner_opt_config = flow.OptConfig(
    n_trials=5,
    throughput=2,
)
inner_loop_config = driver.LayerConfig(
    layer_name='inner_loop',
    universal_params=[few_shot_params, reasoning_param],
    opt_config=inner_opt_config,
)

# ================= Ensemble Options =================
def add_ensemble_option(lm_name):
    usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.9)
    ensemble_param = ensemble.ModuleEnsemble(
        f"ensemble_{lm_name}", [usc_ensemble]
    )
    ensemble_param.module_name = lm_name
    return ensemble_param

ensemble_params = [
    add_ensemble_option('table_selection'),
    add_ensemble_option('candidate_generation'),
    add_ensemble_option('revision'),
]

# ================= Outer Loop Config =================
outer_opt_config = flow.OptConfig(
    n_trials=4,
    throughput=4,
)

outer_loop_config = driver.LayerConfig(
    layer_name='outer_loop',
    dedicate_params=ensemble_params,
    opt_config=outer_opt_config,
)

opt_layer_configs = [inner_loop_config]