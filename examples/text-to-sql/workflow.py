# ⚠️ USE AT YOUR OWN RISK
# first: pip install pysqlite3-binary
# then in settings.py:

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
from datetime import datetime
from src.pipeline.workflow_builder import (
    build_pipeline,
    keyword_extraction,
    entity_retrieval,
    context_retrieval,
    column_filtering,
    table_selection,
    column_selection,
    candidate_generation,
    revision,
    evaluation
)

from src.runner.run_manager import RunManager
from src.runner.task import Task
from src.utils import parse_arguments
import cognify

@cognify.register_workflow
def worker_opt(args, dataset):
    """
    Main function to run the pipeline with the specified configuration.
    """
    assert len(dataset) == 1, "Worker process perform one task at a time"
    
    run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    task = Task(dataset[0])
    result_dir = f"eval_origin/{task.db_id}/{task.question_id}/{run_start_time}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    run_manager = RunManager(args, result_dir)
    run_manager.initialize_tasks(dataset)
    task = run_manager.tasks[0]
    
    result = run_manager.worker(task)
    run_manager.task_done(result, show_progress=False) 

    return {'stats': run_manager.statistics_manager.statistics.to_dict()}

# import litellm
# litellm.set_verbose = True

def worker_demo(query):
    """
    Main function to run the pipeline with the specified configuration.
    """

    full_query = {
        "question_id": 76,
        "db_id": "california_schools",
        "question": "What is the city location of the high school level school with Lunch Provision 2 whose lowest grade is 9 and the highest grade is 12 in the county of Merced?",
        "evidence": "High school can be represented as EILCode = 'HS'",
        "SQL": "SELECT T2.City FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T1.`NSLP Provision Status` = 'Lunch Provision 2' AND T2.County = 'Merced' AND T1.`Low Grade` = 9 AND T1.`High Grade` = 12 AND T2.EILCode = 'HS'",
        "difficulty": "moderate"
    }

    args = parse_arguments()
    args.pipeline_nodes = args.pipeline_nodes.replace("column_filtering+", "")
    args.pipeline_nodes = args.pipeline_nodes.replace("column_selection+", "")
    args.pipeline_nodes = args.pipeline_nodes.replace("table_selection+", "")
    run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    task = Task(full_query)
    result_dir = f"demo_one/{task.db_id}/{task.question_id}/{run_start_time}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    run_manager = RunManager(args, result_dir)
    run_manager.initialize_tasks([full_query])
    task = run_manager.tasks[0]
    
    result = run_manager.worker(task)
    run_manager.task_done(result, show_progress=False) 

    return {'stats': run_manager.statistics_manager.statistics.to_dict()}

