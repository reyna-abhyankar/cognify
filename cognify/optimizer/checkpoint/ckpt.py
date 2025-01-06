import uuid
import json
from typing import Any
import logging
import numpy as np

from cognify.optimizer.utils import _report_cost_reduction, _report_quality_impv
from cognify.optimizer.core.glob_config import GlobalOptConfig
from cognify.optimizer.core.flow import EvaluationResult
from cognify.optimizer.checkpoint import pbar_utils
from cognify.optimizer.evaluator import EvalTask

logger = logging.getLogger(__name__)

class TrialLog:
    def __init__(
        self,
        layer_name: str,
        params: dict[str, any],
        result: EvaluationResult = None,
        id: str = None,
        eval_task_dict: dict = None,
    ):
        self.layer_name = layer_name
        self.id: str = id or uuid.uuid4().hex
        self.params = params
        self.result = result
        self.eval_task_dict = eval_task_dict

    def to_dict(self):
        return {
            "layer_name": self.layer_name,
            "id": self.id,
            "params": self.params,
            "result": self.result.to_dict(),
            "eval_task": self.eval_task_dict,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            layer_name=data["layer_name"],
            id=data["id"],
            params=data["params"],
            result=EvaluationResult.from_dict(data["result"]),
            eval_task_dict=data.get("eval_task"),
        )
        
    def show_transformation(self) -> str:
        eval_task = EvalTask.from_dict(self.eval_task_dict)
        return eval_task.show_opt_trace()

def get_pareto_front(candidates: list[TrialLog]) -> list[TrialLog]:
    """Find the pareto-efficient points

    NOTE: Filter the candidate list before calling this function
    """
    if not candidates:
        return []
    score_cost_list = []
    for trial_log in candidates:
        score_cost_list.append((trial_log.result.reduced_score, trial_log.result.reduced_price))

    vectors = np.array([[-score, price] for score, price in score_cost_list])
    is_efficient = np.ones(vectors.shape[0], dtype=bool)
    for i, v in enumerate(vectors):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(
                vectors[is_efficient] < v, axis=1
            )  # Keep any point with a lower cost
            is_efficient[i] = True  # And keep self

    # return filtered [T_ParetoProgram]
    pareto_frontier = [
        log for log, eff in zip(candidates, is_efficient) if eff
    ]
    return pareto_frontier
    
class LayerStat:
    def __init__(
        self, 
        layer_name: str, 
        instance_id: str, 
        opt_log_path: str, 
        is_leaf: bool
    ):
        self.layer_name = layer_name
        self.instance_id = instance_id
        self.opt_logs: dict[str, TrialLog] = {}
        self.best_score = None
        self.lowest_cost = None
        self.opt_cost = 0.0
        self.opt_log_path = opt_log_path
        self.is_leaf = is_leaf
    
    def add_trial(self, params) -> str:
        _id = self._inc_log_id()
        log = TrialLog(layer_name=self.layer_name, params=params, id=_id)
        self.opt_logs[log.id] = log
        return _id
    
    def _inc_log_id(self):
        max_id = None
        for log in self.opt_logs.values():
            id = int(log.id.split("_")[-1])
            max_id = id if max_id is None else max(max_id, id)
        new_trial_number = max_id + 1 if max_id is not None else 0
        return f"{self.instance_id}_{new_trial_number}"
    
    def report_result(self, id: str, result: EvaluationResult):
        self.opt_logs[id].result = result
        self.opt_cost += result.total_eval_cost
        score, price = result.reduced_score, result.reduced_price
        if self.best_score is None or score > self.best_score:
            self.best_score = score
        if self.lowest_cost is None or price < self.lowest_cost:
            self.lowest_cost = price

        logger.debug(
            f"- {self.instance_id} - Trial id {id} result: score= {score:.2f}, cost@1000= {price*1000:.3f}"
        )
            
        pbar_utils.add_opt_progress(
            name=self.instance_id,
            score=self.best_score,
            price=self.lowest_cost,
            total_cost=self.opt_cost,
            is_evaluator=False,
        )
    
    def load_existing_logs(self):
        with open(self.opt_log_path, "r") as f:
            opt_trace = json.load(f)
        for trial_log_id, trial_meta in opt_trace.items():
            trial_log = TrialLog.from_dict(trial_meta)
            self.opt_logs[trial_log_id] = trial_log
            self.opt_cost += trial_log.result.total_eval_cost
            
        cancidates = self.get_all_candidates()
        if cancidates:
            self.best_score = max([log.result.reduced_score for log in cancidates])
            self.lowest_cost = min([log.result.reduced_price for log in cancidates])
    
    def save_opt_logs(self):
        opt_logs_json_obj = {}
        for k, v in self.opt_logs.items():
            if v.result and v.result.complete:
                opt_logs_json_obj[k] = v.to_dict()
        if not opt_logs_json_obj:
            logger.warning("No finished trials to save")
            return
        json.dump(opt_logs_json_obj, open(self.opt_log_path, "w+"), indent=4)
    
    def get_opt_summary(self):
        candidates = self.get_all_candidates()
        pareto_frontier = get_pareto_front(candidates)
        return pareto_frontier

    @property
    def all_finished(self):
        return all([log.result.complete for log in self.opt_logs.values()])
    
    def init_progress_bar(
        self,
        level: int,
        budget: int,
        leave: bool,
    ):
        initial = len(self.opt_logs)
        total = initial + budget
        initial_desc = pbar_utils._gen_opt_bar_desc(
            self.best_score,
            self.lowest_cost,
            self.opt_cost,
            self.instance_id,
            level + 1,
        )
        pbar_utils.add_pbar(
            name=self.instance_id,
            desc=initial_desc,
            total=total,
            initial=initial,
            leave=leave,
            indent=level + 1,
        )
    
    def get_completed_logs(self) -> list[TrialLog]:
        return [log for log in self.opt_logs.values() if log.result.complete]
    
    def get_all_candidates(self) -> list[TrialLog]:
        """get logs that are qualified
        """
        cancidates = []
        for log_id, log in self.opt_logs.items():
            if not log.result.complete:
                continue
            # if not meet the quality constraint, skip
            if (
                GlobalOptConfig.quality_constraint is not None
                and log.result.reduced_score < GlobalOptConfig.quality_constraint
            ):
                continue
            cancidates.append(log)
        return cancidates
            
    
class LogManager:
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LogManager, cls).__new__(cls)
            cls._instance._init(*args, **kwargs)
        return cls._instance
    
    def _init(self, base_score, base_cost):
        self.layer_stats: dict[str, LayerStat] = {}
        self._glob_best_score = base_score
        self._glob_lowest_cost = base_cost
    
    def register_layer(
        self, 
        layer_name: str,
        layer_instance: str, 
        opt_log_path: str, 
        is_leaf: bool
    ):
        if layer_instance in self.layer_stats:
            raise ValueError(f"Layer {layer_instance} already registered at LogManager")
        self.layer_stats[layer_instance] = LayerStat(layer_name, layer_instance, opt_log_path, is_leaf)
    
    def add_trial(self, layer_instance: str, params) -> str:
        return self.layer_stats[layer_instance].add_trial(params)
    
    def report_trial_result(self, layer_instance: str, id: str, result: EvaluationResult):
        if result is None or not result.complete:
            return
        self.layer_stats[layer_instance].report_result(id, result)
        self._update_glob_best_score(result.reduced_score)
        self._update_glob_lowest_cost(result.reduced_price)
    
    def load_existing_logs(self, layer_instance: str):
        self.layer_stats[layer_instance].load_existing_logs()
        # update global best score and lowest cost
        local_best_score = self.layer_stats[layer_instance].best_score
        local_lowest_cost = self.layer_stats[layer_instance].lowest_cost
        self._update_glob_best_score(local_best_score)
        self._update_glob_lowest_cost(local_lowest_cost)
    
    def get_global_summary(self, verbose: bool):
        candidates = []
        for layer_instance, layer_stat in self.layer_stats.items():
            if layer_stat.is_leaf:
                candidates.extend(layer_stat.get_all_candidates())
        pareto_frontier = get_pareto_front(candidates)
        if verbose:
            print(f"================ Optimization Results =================") 
            print(f"Num Pareto Frontier: {len(pareto_frontier)}")
            for i, trial_log in enumerate(pareto_frontier):
                print("--------------------------------------------------------")
                print("Pareto_{}".format(i + 1))
                score, price = trial_log.result.reduced_score, trial_log.result.reduced_price
                # logger.info("  Params: {}".format(trial_log.params))
                if GlobalOptConfig.base_quality is not None:
                    print("  Quality improves by {:.0f}%".format(_report_quality_impv(score, GlobalOptConfig.base_quality)))
                if GlobalOptConfig.base_price is not None:
                    print("  Cost is {:.2f}x original".format(_report_cost_reduction(price, GlobalOptConfig.base_price)))
                print("  Quality: {:.2f}, Cost per 1K invocation: ${:.2f}".format(score, price * 1000))
                # print("  Applied at: {}".format(trial_log.id))
                # logger.info("  config saved at: {}".format(log_path))

            # logger.info("Opt Cost: {}".format(self.opt_cost))
            print("========================================================")
        
        finished_trials: dict[str, tuple[TrialLog, str]] = {}
        total_opt_cost = 0
        for layer_instance, layer_stat in self.layer_stats.items():
            if layer_stat.is_leaf:
                total_opt_cost += layer_stat.opt_cost
                for log in layer_stat.get_completed_logs():
                    finished_trials[log.id] = (log, layer_stat.opt_log_path)
        
        return total_opt_cost, pareto_frontier, finished_trials

    def get_log_by_id(self, log_id: str) -> TrialLog:
        layer_instance = log_id.rsplit("_", 1)[0]
        return self.layer_stats[layer_instance].opt_logs[log_id]
        
    def _update_glob_best_score(self, score):
        if score is None:
            return
        if self._glob_best_score is None or score > self._glob_best_score:
            self._glob_best_score = score
    
    def _update_glob_lowest_cost(self, cost):
        if cost is None:
            return
        if self._glob_lowest_cost is None or cost < self._glob_lowest_cost:
            self._glob_lowest_cost = cost
    