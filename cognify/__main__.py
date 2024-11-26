import argparse
import debugpy
import logging


from cognify.cognify_args import (
    init_cognify_args,
    OptimizationArgs,
    EvaluationArgs,
    InspectionArgs,
)
from cognify.optimizer.plugin import capture_module_from_fs
from cognify.optimizer.registry import get_registered_data_loader
from cognify.optimizer.control_param import ControlParameter
from cognify.run.optimize import optimize
from cognify.run.evaluate import evaluate
from cognify.run.inspect import inspect
from cognify._logging import _configure_logger

logger = logging.getLogger(__name__)


def from_cognify_args(args):
    if args.mode == "optimize":
        return OptimizationArgs.from_cli_args(args)
    elif args.mode == "evaluate":
        return EvaluationArgs.from_cli_args(args)
    elif args.mode == "inspect":
        return InspectionArgs.from_cli_args(args)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


def parse_pipeline_config_file(config_path, load_data: bool = True):
    config_module = capture_module_from_fs(config_path)

    # get optimizer control parameters
    control_param = ControlParameter.build_control_param(loaded_module=config_module)
    if not load_data:
        return None, control_param

    # load data
    data_loader_fn = get_registered_data_loader()
    train_set, val_set, test_set = data_loader_fn()
    logger.info(
        f"size of train set: {0 if not train_set else len(train_set)}, "
        f"val set: {0 if not val_set else len(val_set)}, "
        f"test set: {0 if not test_set else len(test_set)}"
    )

    return (train_set, val_set, test_set), control_param


def optimize_routine(opt_args: OptimizationArgs):
    (train_set, val_set, test_set), control_param = parse_pipeline_config_file(
        opt_args.config
    )

    cost, frontier, opt_logs = optimize(
        script_path=opt_args.workflow,
        control_param=control_param,
        train_set=train_set,
        val_set=val_set,
        eval_fn=None,
        eval_path=opt_args.config,
        resume=opt_args.resume,
        force=opt_args.force,
    )
    return cost, frontier, opt_logs


def evaluate_routine(eval_args: EvaluationArgs):
    (train_set, val_set, test_set), control_param = parse_pipeline_config_file(
        eval_args.config
    )
    result = evaluate(
        config_id=eval_args.select,
        test_set=test_set,
        workflow=eval_args.workflow,
        n_parallel=eval_args.n_parallel,
        eval_fn=None,
        eval_path=eval_args.config,
        save_to=eval_args.output_path,
        control_param=control_param,
    )
    return result


def inspect_routine(inspect_args: InspectionArgs):
    _, control_param = parse_pipeline_config_file(inspect_args.config, load_data=False)
    inspect(
        control_param=control_param,
        dump_frontier_details=inspect_args.dump_frontier_details,
    )


def main():
    # debugpy.listen(5678)
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    # debugpy.breakpoint()

    parser = argparse.ArgumentParser()
    init_cognify_args(parser)
    raw_args = parser.parse_args()
    _configure_logger(raw_args.log_level)

    cognify_args = from_cognify_args(raw_args)
    if raw_args.mode == "optimize":
        optimize_routine(cognify_args)
    elif raw_args.mode == "evaluate":
        evaluate_routine(cognify_args)
    else:
        inspect_routine(cognify_args)
    return


if __name__ == "__main__":
    main()