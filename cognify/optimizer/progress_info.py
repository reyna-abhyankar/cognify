import threading
import heapq
from termcolor import colored
import sys

from tqdm.auto import tqdm

class ProgressInfo:

    pbar_lock = threading.Lock()
    max_position = 20
    position_pool = list(range(max_position))
    heapq.heapify(position_pool)

    def __init__(self):
        # self.pbar = tqdm()
        self.best_score = float('-inf')
        self.lowest_cost = float('inf')
        self.lowest_exec_time = float('inf')

    def init_pbar(self, total, initial, initial_score, initial_cost, initial_exec_time, opt_cost):
        self.pbar_position = ProgressInfo.ask_for_position()
        self.pbar = tqdm(
            total=total,
            initial=initial,
            desc=self._gen_opt_bar_desc(initial_score, initial_cost, initial_exec_time, opt_cost),
            leave=True,
            position=self.pbar_position,
            colour="green",
            bar_format=r'{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}]'
        )


    @staticmethod
    def ask_for_position():
        with ProgressInfo.pbar_lock:
            # if no position is available, add a new one
            if len(ProgressInfo.position_pool) == 0:
                position = ProgressInfo.max_position
                ProgressInfo.max_position += 1
            else:
                position = heapq.heappop(ProgressInfo.position_pool)
            return position

    @staticmethod
    def release_position(position):
        with ProgressInfo.pbar_lock:
            heapq.heappush(ProgressInfo.position_pool, position)

    def _gen_pbar_desc(self, level, tb, score, price, exec_time):
        indent = "---" * level + ">"
        color = "green"
        score_text = colored(f"{score:.2f}", color)
        cost_text = colored(f"${price*1000:.2f}", color)
        exec_time_text = colored(f"{exec_time:.2f}s", color)
        return f"{indent} Evaluation in {tb} | (avg score: {score_text}, avg cost@1000: {cost_text}, avg execution time: {exec_time_text})"


    def _gen_opt_bar_desc(self, score, cost, exec_time, total_opt_cost):
        # indent = "---" * hierarchy_level + ">"
        color = "green"
        score = score or 0.0
        cost = cost or 0.0
        exec_time = exec_time or 0.0
        score_text = colored(f"{score:.2f}", color)
        cost_text = colored(f"${cost*1000:.2f}", color)
        exec_time_text = colored(f"{exec_time:.2f}s", color)
        total_opt_cost_text = colored(f"${total_opt_cost:.2f}", color)

        return f"Optimization progress | best score: {score_text}, lowest cost@1000: {cost_text}, lowest exec time: {exec_time_text} | Total Optimization Cost: {total_opt_cost_text}"

    def update_progress(self, frac: float):
        with ProgressInfo.pbar_lock:
            self.pbar.update(frac)

    def update_status(self, best_score, lowest_cost, lowest_exec_time, opt_cost):
        with ProgressInfo.pbar_lock:
            if best_score is not None:
                self.best_score = max(best_score, self.best_score)

            if lowest_cost is not None:
                self.lowest_cost = min(lowest_cost, self.lowest_cost)
                
            if lowest_exec_time is not None:
                self.lowest_exec_time = min(lowest_exec_time, self.lowest_exec_time)

            best_score_desc = 0.0 if self.best_score == float('-inf') else self.best_score
            lowest_cost_desc = 0.0 if self.lowest_cost == float('inf') else self.lowest_cost
            lowest_exec_time_desc = 0.0 if self.lowest_exec_time == float('inf') else self.lowest_exec_time

            self.pbar.set_description(
                self._gen_opt_bar_desc(best_score_desc, lowest_cost_desc, lowest_exec_time_desc, opt_cost)
            )
            self.pbar.update(0)

    def release(self, hierarchy_level):
        if hierarchy_level == 0:
            ProgressInfo.release_position(self.pbar_position)

pbar = ProgressInfo()