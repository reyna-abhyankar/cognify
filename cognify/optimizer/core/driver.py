import os
import json
from typing import Union, Optional, Callable
import logging
import re

from cognify.optimizer.evaluator import (
    EvaluationResult,
    EvaluatorPlugin,
    EvalTask,
)
from cognify.optimizer.core.flow import TopDownInformation, LayerConfig
from cognify.optimizer.core.opt_layer import OptLayer, GlobalOptConfig
from cognify.optimizer.checkpoint.ckpt import LogManager, TrialLog
from cognify.optimizer.utils import _report_cost_reduction, _report_quality_impv, _report_exec_time_reduction
from cognify.optimizer.utils import _report_quality_impv_raw, _report_cost_reduction_raw, _report_exec_time_reduction_raw

from cognify.optimizer.control_param import SelectedObjectives
from cognify._tracing import trace_quality_improvement, trace_cost_improvement, trace_latency_improvement

logger = logging.getLogger(__name__)

def get_layer_evaluator_factory(
    next_layer_factory, 
    layer_config: LayerConfig,
    level: int,
    is_leaf: bool,
    selected_objectives: SelectedObjectives
):
    def _factory():
        return OptLayer(
            name=layer_config.layer_name,
            opt_config=layer_config.opt_config,
            objectives=selected_objectives,
            hierarchy_level=level,
            is_leaf=is_leaf,
            next_layer_factory=next_layer_factory,
            dedicate_params=layer_config.dedicate_params,
            universal_params=layer_config.universal_params,
            target_modules=layer_config.target_modules,
        )
    return _factory

class MultiLayerOptimizationDriver:
    def __init__(
        self,
        layer_configs: list[LayerConfig],
        opt_log_dir: str,
        objectives: SelectedObjectives,
        quality_constraint: float = None,
        base_quality: float = None,
        base_cost: float = None,
        base_exec_time: float = None,
    ):
        """Driver for multi-layer optimization

        Args:
            layer_configs (Sequence[LayerConfig]): configs for each optimization layer

        NOTE: the order of the layers is from top to bottom, i.e., the last layer will run program evaluation directly while others will run layer evaluation
        """
        self.layer_configs = layer_configs
        self.objectives = objectives
        
        GlobalOptConfig.quality_constraint = quality_constraint
        GlobalOptConfig.base_quality = base_quality
        GlobalOptConfig.base_price = base_cost
        GlobalOptConfig.base_exec_time = base_exec_time
        _log_mng = LogManager(base_quality, base_cost, base_exec_time)

        # initialize optimization layers
        self.opt_layer_factories: list[Callable] = [None] * (len(layer_configs) + 1)

        self.opt_log_dir = opt_log_dir

        # config log dir for layer opts
        # NOTE: only the top layer will be set, others are decided at runtime
        self.layer_configs[0].opt_config.log_dir = os.path.join(
            opt_log_dir, self.layer_configs[0].layer_name
        )
        # NOTE: since these will be set at runtime, we set them to None
        for layer_config in self.layer_configs[1:]:
            layer_config.opt_config.log_dir = None
            layer_config.opt_config.opt_log_path = None
            layer_config.opt_config.param_save_path = None
            
    def build_tiered_optimization(self, evaluator: EvaluatorPlugin):
        """Build tiered optimization from bottom to top"""
        self.opt_layer_factories[-1] = lambda: evaluator
        
        for ri, layer_config in enumerate(reversed(self.layer_configs)):
            idx = len(self.layer_configs) - ri - 1
            next_layer_factory = self.opt_layer_factories[idx + 1]
            current_layer_factory = get_layer_evaluator_factory(next_layer_factory, layer_config, idx, ri == 0, selected_objectives=self.objectives)
            self.opt_layer_factories[idx] = current_layer_factory

    def run(
        self,
        evaluator: EvaluatorPlugin,
        script_path: str,
        script_args: Optional[list[str]] = None,
        other_python_paths: Optional[list[str]] = None,
    ) -> tuple[float, list[tuple[TrialLog, str]], dict[str, TrialLog]]:
        self.build_tiered_optimization(evaluator)
        top_layer = self.opt_layer_factories[0]()
        logger.info("----------------- Start Optimization -----------------")
        opt_cost, frontier, finished_opt_logs = top_layer.easy_optimize(
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
        logger.info("----------------- Optimization Finished -----------------")
        self.dump_frontier_details(frontier, finished_opt_logs)
        return opt_cost, frontier, finished_opt_logs

    def _extract_trial_id(self, config_id: str) -> str:
        param_log_dir = os.path.join(self.opt_log_dir, "optimized_workflow_details")
        if not os.path.exists(param_log_dir):
            raise ValueError(
                f"Cannot find the optimization log directory at {param_log_dir}"
            )

        with open(os.path.join(param_log_dir, f"{config_id}.cog"), "r") as f:
            first_line = f.readline().strip()
        match = re.search(r"Trial - (.+)", first_line)
        if match:
            trial_id = match.group(1)
            return trial_id
        else:
            raise ValueError(
                f"Cannot extract trial id from the log file {config_id}.cog"
            )

    def _find_config_log_path(self, trial_id: str) -> str:
        opt_config = self.layer_configs[0].opt_config
        opt_config.finalize()
        tdi = TopDownInformation(
            opt_config=opt_config,
            all_params=None,
            module_ttrace=None,
            current_module_pool=None,
            script_path=None,
            script_args=None,
            other_python_paths=None,
        )

        top_layer = self.opt_layers[0]
        top_layer.load_opt_log(opt_config.opt_log_path)
        top_layer.top_down_info = tdi
        all_configs = top_layer.get_all_candidates()
        config_path = None

        for opt_log, path in all_configs:
            if opt_log.id == trial_id:
                config_path = path
                break
        else:
            raise ValueError(f"Config {trial_id} not found in the optimization log.")
        return config_path

    def evaluate(
        self,
        evaluator: EvaluatorPlugin,
        config_id: str,
    ) -> EvaluationResult:
        self.load_from_file()
        trial_id = self._extract_trial_id(config_id)
        log = LogManager().get_log_by_id(trial_id)

        # apply selected trial
        print(f"----- Testing {config_id} -----")
        # print("  Training Quality: {:.3f}, Cost per 1K invocation: ${:.2f}\n".format(trial_log.score, trial_log.price * 1000))
        
        eval_task = EvalTask.from_dict(log.eval_task_dict)
        # run evaluation
        eval_result = evaluator.get_score(mode='test', task=eval_task, show_process=True, keep_bar=True)
        
        print(f"=========== Evaluation Results ===========") 
        if GlobalOptConfig.base_quality is not None:
            print(_report_quality_impv(eval_result.reduced_score, GlobalOptConfig.base_quality))
        if GlobalOptConfig.base_price is not None:
            print(_report_cost_reduction(eval_result.reduced_price, GlobalOptConfig.base_price))
        if GlobalOptConfig.base_exec_time is not None:
            print(_report_exec_time_reduction(eval_result.reduced_exec_time, GlobalOptConfig.base_exec_time))
        print("  Quality: {:.2f}, Cost per 1K invocation: ${:.2f}, Avg exec time: {:.2f} s".format(eval_result.reduced_score, eval_result.reduced_price * 1000, eval_result.reduced_exec_time))
        print("===========================================")

        return eval_result

    def load(
        self,
        config_id: str,
    ):
        self.load_from_file()
        trial_id = self._extract_trial_id(config_id)
        log = LogManager().get_log_by_id(trial_id)
        eval_task = EvalTask.from_dict(log.eval_task_dict)
        schema, old_name_2_new_module = eval_task.load_and_transform()
        return schema, old_name_2_new_module

    def inspect(self, dump_details: bool = False):
        self.load_from_file()
        # dump frontier details to file
        opt_cost, pareto_frontier, finished_opt_logs = LogManager().get_global_summary(verbose=True, selected_objectives=self.objectives)
        if dump_details:
            self.dump_frontier_details(pareto_frontier, finished_opt_logs)
        return

    def dump_frontier_details(self, frontier, finished_opt_logs):
        param_log_dir = os.path.join(self.opt_log_dir, "optimized_workflow_details")
        if not os.path.exists(param_log_dir):
            os.makedirs(param_log_dir, exist_ok=True)
        for i, trial_log in enumerate(frontier):
            trial_log: TrialLog
            score, price, exec_time = trial_log.result.reduced_score, trial_log.result.reduced_price, trial_log.result.reduced_exec_time
            dump_path = os.path.join(param_log_dir, f"Optimization_{i+1}.cog")
            details = f"Trial - {trial_log.id}\n"
            details += f"Optimized for {str(self.objectives)}\n"
            log_path = finished_opt_logs[trial_log.id][1]
            details += f"Log at: {log_path}\n"
            if GlobalOptConfig.base_quality is not None:
                details += _report_quality_impv(score, GlobalOptConfig.base_quality)
                trace_quality_improvement(_report_quality_impv_raw(score, GlobalOptConfig.base_quality))
            if GlobalOptConfig.base_price is not None:
                details += _report_cost_reduction(price, GlobalOptConfig.base_price)
                trace_cost_improvement(_report_cost_reduction_raw(score, GlobalOptConfig.base_quality))
            if GlobalOptConfig.base_exec_time is not None:
                details += _report_exec_time_reduction(exec_time, GlobalOptConfig.base_exec_time)
                trace_latency_improvement(_report_exec_time_reduction_raw(score, GlobalOptConfig.base_quality))
                
            details += f"Quality: {score:.3f}, Cost per 1K invocation: ${price * 1000:.2f}, Avg exec time: {exec_time:.2f} s\n"
            trans = trial_log.show_transformation()
            details += trans
            with open(dump_path, "w") as f:
                f.write(details)
    
    def load_from_file(self):
        root_log = self.layer_configs[0].opt_config
        root_log.finalize()
        _log_dir_stack = [root_log.log_dir]
        leaf_layer_name = self.layer_configs[-1].layer_name
        
        while _log_dir_stack:
            log_dir = _log_dir_stack.pop()
            opt_log_path = os.path.join(log_dir, "opt_logs.json")
            
            if not os.path.exists(opt_log_path):
                continue
            with open(opt_log_path, "r") as f:
                opt_trace = json.load(f)
                
                for log_id, log in opt_trace.items():
                    layer_instance = log_id.rsplit("_", 1)[0]
                    layer_name = log["layer_name"]
                    trial_number = log_id.rsplit("_", 1)[-1]
                    sub_layer_log_dir = os.path.join(log_dir, f"{layer_name}_trial_{trial_number}")
                    _log_dir_stack.append(sub_layer_log_dir)
                    
                LogManager().register_layer(
                    layer_name=layer_name,
                    layer_instance=layer_instance,
                    opt_log_path=opt_log_path,
                    is_leaf=layer_name == leaf_layer_name,
                )
                LogManager().load_existing_logs(layer_instance)
            