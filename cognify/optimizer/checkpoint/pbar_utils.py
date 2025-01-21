import heapq
import threading
from cognify.optimizer.utils import _cognify_tqdm as tqdm
from cognify.optimizer.core.flow import EvaluationResult

_max_position = 20
_position_pool = list(range(_max_position))
heapq.heapify(_position_pool)
_pbar_lock = threading.Lock()

def ask_for_position():
    global _max_position
    with _pbar_lock:
        # if no position is available, add a new one
        if len(_position_pool) == 0:
            position = _max_position
            _max_position += 1
        else:
            position = heapq.heappop(_position_pool)
        return position


def release_position(position):
    with _pbar_lock:
        heapq.heappush(_position_pool, position)

_pbar_pool: dict[str, tuple[int, int, tqdm]] = {}

def _gen_opt_bar_desc(
    best_score, 
    lowest_price,
    fastest_time,
    opt_cost,
    name,
    indent,
    /,
):
    prefix = "---" * indent + ">"
    formatted_best_score = f"{best_score:.2f}" if best_score is not None else "N/A"
    formatted_lowest_price = f"${lowest_price*1000:.2f}" if lowest_price is not None else "N/A"
    formatted_fastest_time = f"{fastest_time:.2f} s" if fastest_time is not None else "N/A"
    return f"{prefix} {name} | (best score: {formatted_best_score}, lowest cost@1000: {formatted_lowest_price}, fastest workflow: avg {formatted_fastest_time}) | Total Optimization Cost: ${opt_cost:.2f}"

def _gen_eval_bar_desc(
    score, 
    price,
    exec_time,
    eval_cost,
    name,
    indent,
    /,
):
    prefix = "---" * indent + ">"
    return f"{prefix} {name} | (avg score: {score:.2f}, avg cost@1000: ${price*1000:.2f}, avg exec: {exec_time:.2f} s) | Total Evaluation Cost: ${eval_cost:.2f}"

def add_pbar(
    name: str, 
    desc: str, 
    total: int, 
    initial: int,
    leave: bool,
    indent: int
):
    if name in _pbar_pool:
        raise ValueError(f"pbar with name {name} already exists")
    position = ask_for_position()
    pbar = tqdm(
        desc=desc,
        total=total,
        initial=initial,
        position=position,
        leave=leave,
    )
    _pbar_pool[name] = (indent, position, pbar)

def close_pbar(name: str):
    if name not in _pbar_pool:
        raise ValueError(f"pbar with name {name} does not exist")
    _, pos, pbar = _pbar_pool[name]
    pbar.close()
    release_position(pos)
    del _pbar_pool[name]

def add_opt_progress(
    name: str,
    score: float,
    price: float,
    exec_time: float,
    total_cost: float,
    is_evaluator: bool,
):
    indent, _, pbar = _pbar_pool[name]
    _gen_desc = _gen_eval_bar_desc if is_evaluator else _gen_opt_bar_desc
    desc = _gen_desc(score, price, exec_time, total_cost, name, indent)
    pbar.set_description(desc)
    pbar.update(1)