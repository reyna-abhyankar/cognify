import os
from typing import Optional, Callable
import logging
from cognify.optimizer.control_param import ControlParameter
from cognify.optimizer.core import driver
from cognify.optimizer.plugin import OptimizerSchema
from cognify._signal import _should_exit, _init_exit_gracefully

_init_exit_gracefully(msg="Stopping main", verbose=True)

logger = logging.getLogger(__name__)

def run(
    *,
    config_id: str,
    workflow: str,
    input: str,
    control_param: Optional[ControlParameter] = None,
):
    if config_id == 'NoChange':
        # print("Loading original workflow")
        program = OptimizerSchema.capture(workflow).program
    else:
        # print(f"Loading workflow with {config_id}")
        program = load_workflow(config_id=config_id, control_param=control_param)
    # print(f"Running the following input: '{input}'...")
    
    text2sql_query = {
        "question_id": "76",
        "db_id": "california_schools",
        "question": input,
        "evidence": "High school can be represented as EILCode = 'HS'",
        "SQL": "SELECT T2.City FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`NSLP Provision Status` = 'Lunch Provision 2' AND T2.County = 'Merced' AND T1.`Low Grade` = 9 AND T1.`High Grade` = 12 AND T2.EILCode = 'HS'",
        "difficulty": "moderate"
    }
    input = {"query": text2sql_query}
    result = program(**input)
    # print(f"Output: {result.values()[0]}")
    
def load_workflow(
    *,
    config_id: str,
    opt_result_path: Optional[str] = None,
    control_param: Optional[ControlParameter] = None,
) -> Callable:
    assert (
        control_param or opt_result_path
    ), "Either control_param or opt_result_path should be provided"
    # If both are provided, control_param will be used
    if control_param is None:
        control_param_save_path = os.path.join(opt_result_path, "control_param.json")
        control_param = ControlParameter.from_json_profile(control_param_save_path)
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=control_param.opt_layer_configs,
        opt_log_dir=control_param.opt_history_log_dir,
    )
    schema, _ = opt_driver.load(config_id)
    return schema.program