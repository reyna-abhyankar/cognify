import importlib.util
from typing import Callable, Any
import importlib
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

from cognify.graph.program import Module
from cognify.optimizer.registry import (
    clear_registry,
    get_registered_opt_program_entry,
    get_registered_opt_modules,
)


class OptimizerSchema:
    def __init__(
        self,
        program: Callable[[Any], Any],
        opt_target_modules: list[Module],
    ):
        self.program = program
        self.opt_target_modules = opt_target_modules
        # logger.info(f"modules cap: {opt_target_modules}")

    @classmethod
    def capture(cls, script_path: str) -> "OptimizerSchema":
        clear_registry()
        capture_module_from_fs(script_path)
        opt_targets = get_registered_opt_modules()
        if not opt_targets:
            logger.warning("No Optimizable modules found")
        program = get_registered_opt_program_entry()
        if not program:
            raise ValueError("The workflow entry point is not defined, please register it with `@register_workflow`")
        schema = cls(
            program=program,
            opt_target_modules=opt_targets,
        )
        return schema


def capture_module_from_fs(module_path: str):
    logger.debug(f"obtain module at: {module_path}")

    try:
        path = Path(module_path)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        logger.error(f"Failed to load module from {module_path}")
        raise

    # translate
    num_translated = 0
    named_runnables = defaultdict(int)
    
    # lazy import
    import dspy
    from cognify.frontends.dspy.connector import PredictModel
    from cognify.frontends.langchain.connector import RunnableModel
    from langchain_core.runnables import RunnableSequence
    
    # check if user has manually wrapped their runnables
    is_manually_translated = False
    for k, v in module.__dict__.items():
        if isinstance(v, RunnableModel) or isinstance(v, PredictModel):
            is_manually_translated = True
            break

    # translate
    if not is_manually_translated:
        named_runnables = defaultdict(int)
        for k, v in module.__dict__.items():
            if isinstance(v, dspy.Module):
                named_predictors = v.named_predictors()
                for name, predictor in named_predictors:
                    module.__dict__[k].__dict__[name] = PredictModel(name, predictor)
                    num_translated += 1
            elif isinstance(v, RunnableSequence):
                # ensure unique naming for runnable
                name = k if named_runnables[k] == 0 else f"{k}_{named_runnables[k]}"
                module.__dict__[k] = RunnableModel(name, v)
                named_runnables[k] += 1

    return module
