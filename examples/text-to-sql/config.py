import argparse
import json
from datetime import datetime
import os
import debugpy
import multiprocessing as mp
import sys
from src.utils import parse_arguments, read_from_file

import cognify
import numpy as np
import warnings


@cognify.register_data_loader
def load_data():
    args = parse_arguments()
    args.pipeline_nodes = args.pipeline_nodes.replace("column_filtering+", "")
    args.pipeline_nodes = args.pipeline_nodes.replace("column_selection+", "")
    args.pipeline_nodes = args.pipeline_nodes.replace("table_selection+", "")
    all_train = read_from_file('data/dev/ca_school.json', args)
    test_set = read_from_file('data/dev/sub_sampled_bird_dev_set.json', args)
    
    # shuffle the data
    # all_train = np.random.permutation(all_train).tolist()
    # return all_train[:100], all_train[100:], test_set[:10]
    return all_train, None, test_set


@cognify.register_evaluator
def eval(stats):
    """
    Evaluate the statistics of the run.
    """
    correct = any(vs['correct'] == 1 for vs in stats['counts'].values())
    return 1.0 if correct else 0.0

# from cognify.hub.search import text_to_sql
# search_settings = text_to_sql.create_search(opt_log_dir="cognify_opt_debit_card", evaluator_batch_size=30)

from cognify.hub.search import default
search_settings = default.create_search(
    search_type='light',
    n_trials=5,
    opt_log_dir='ca_school_opt_demo_5_trial_record_xxxxxxx',
    evaluator_batch_size=40,
)
