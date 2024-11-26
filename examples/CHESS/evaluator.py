from cognify.optimizer import register_workflow, register_evaluator

@register_evaluator
def eval(label, stats):
    """
    Evaluate the statistics of the run.
    """
    correct = any(vs['correct'] == 1 for vs in stats['counts'].values())
    return 1.0 if correct else 0.0