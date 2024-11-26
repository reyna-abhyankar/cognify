# ⚠️ USE AT YOUR OWN RISK
# first: pip install pysqlite3-binary
# then in settings.py:

# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import debugpy
import argparse
import json
from datetime import datetime

from runner.run_manager import RunManager
from runner.database_manager import DatabaseManager
from runner.logger import Logger
from pipeline.pipeline_manager import PipelineManager
from runner.task import Task
from typing import Any, Dict, List, TypedDict, Callable
from langgraph.graph import END, StateGraph

from pipeline.keyword_extraction import keyword_extraction
from pipeline.entity_retrieval import entity_retrieval
from pipeline.context_retrieval import context_retrieval
from pipeline.column_filtering import column_filtering
from pipeline.table_selection import table_selection
from pipeline.column_selection import column_selection
from pipeline.candidate_generation import candidate_generation
from pipeline.revision import revision
from pipeline.evaluation import evaluation
from pipeline.annotated import cognify_registry
from pipeline.workflow_builder import build_pipeline

from cognify.optimizer import register_workflow


@register_workflow
def worker(input):
    """
    Main function to run the pipeline with the specified configuration.
    """
    args = input['args']
    dataset = input['dataset']
    assert len(dataset) == 1, "Worker process perform one task at a time"
    
    run_start_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    task = Task(dataset[0])
    result_dir = f"results/{task.db_id}/{task.question_id}/{run_start_time}"
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

    run_manager = RunManager(args, result_dir)
    run_manager.initialize_tasks(dataset)
    task = run_manager.tasks[0]
    
    result = run_manager.worker(task)
    run_manager.task_done(result, show_progress=False) 

    return run_manager.statistics_manager.statistics.to_dict()

