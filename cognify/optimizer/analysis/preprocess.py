from typing import Callable

from cognify.optimizer.analysis.domain import TextEmbeddingDomainManager, DomainManagerInterface
from cognify.optimizer.control_param import ControlParameter
from cognify.optimizer.core import driver
from cognify.optimizer.evaluator import (
    EvaluatorPlugin,
    EvalTask,
    EvaluationResult,
)

def prepare_input_env(
    script_path: str,
    control_param: ControlParameter,
    train_set,
    eval_fn: Callable = None,
    eval_path: str = None,
    resume: bool = False,
    force: bool = False,
):
    # split dataset to domains
    if control_param.cluster_K > 1:
        
    ...
    