import pandas as pd
import cognify
import dotenv
from cognify.hub.evaluators import f1_score_str
import string
import unicodedata
import json
import re 
dotenv.load_dotenv()

def load_sentiment_data(pool):
    sentiment_df = pd.read_parquet("data/sentiment.parquet")
    data = []
    for i, row in sentiment_df.iterrows():
        input = {
            'task': row['instruction'] + "\n" + row['input'],
            'mode': 'sentiment_analysis'
        }
        output = {
            'label': row['output']
        }
        data.append((input, output))
        if i >= 99:
            break
    pool[0].extend(data)
    pool[2].extend(data)

def load_relation_extraction_data(pool):
    relation_df = pd.read_parquet("data/relation.parquet")
    data = []
    for i, row in relation_df.iterrows():
        input = {
            'task': row['instruction'] + "\n" + row['input'],
            'mode': 'relation_extraction'
        }
        output = {
            'label': row['output']
        }
        data.append((input, output))
        if i >= 99:
            break
    pool[0].extend(data)
    pool[2].extend(data)
    
def load_headline_data(pool):
    headline_df = pd.read_parquet("data/headline.parquet")
    data = []
    for i, row in headline_df.iterrows():
        input = {
            'task': row['instruction'] + "\n" + row['input'],
            'mode': 'headline_classification'
        }
        output = {
            'label': row['output']
        }
        data.append((input, output))
        if i >= 99:
            break
    pool[0].extend(data)
    pool[2].extend(data)
    
def load_fiqa_data(pool):
    fiqa_df = pd.read_parquet("data/fiqa.parquet")
    data = []
    for i, row in fiqa_df.iterrows():
        input = {
            'task': row['instruction'] + "\n" + row['input'],
            'mode': 'fiqa'
        }
        output = {
            'label': row['output']
        }
        data.append((input, output))
        if i >= 99:
            break
    pool[0].extend(data)
    pool[2].extend(data)


@cognify.register_data_loader
def load_data():
    all_train, all_val, all_test = [], None, []
    load_sentiment_data([all_train, all_val, all_test])
    load_headline_data([all_train, all_val, all_test])
    load_fiqa_data([all_train, all_val, all_test])
    
    return all_train, all_val, all_test


@cognify.register_evaluator
def evaluate(answer, label, mode, task):
    if mode == 'sentiment_analysis':
        return evaluate_sentiment(answer, label)
    elif mode == 'relation_extraction':
        return evaluate_relation_extraction(answer, label)
    elif mode == 'headline_classification':
        return evaluate_headline(answer, label)
    elif mode == 'fiqa':
        return evaluate_fiqa(answer, label, task)
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
def evaluate_dspy(true, prediction, trace=None):
    if true.mode == 'sentiment_analysis':
        return evaluate_sentiment(prediction.solution, true.label)
    elif true.mode == 'relation_extraction':
        return evaluate_relation_extraction(prediction.solution, true.label)
    elif true.mode == 'headline_classification':
        return evaluate_headline(prediction.solution, true.label)
    elif true.mode == 'fiqa':
        return evaluate_fiqa(prediction.solution, true.label, true.task)
    else:
        raise ValueError(f"Invalid mode: {true.mode}")

def evaluate_trace(true, prediction, trace=None):
    if true.mode == 'sentiment_analysis':
        return evaluate_sentiment(prediction.solution, true.label)
    elif true.mode == 'relation_extraction':
        return evaluate_relation_extraction(prediction.solution, true.label)
    elif true.mode == 'headline_classification':
        return evaluate_headline(prediction.solution, true.label)
    elif true.mode == 'fiqa':
        return evaluate_fiqa(prediction.solution, true.label, true.task)
    else:
        raise ValueError(f"Invalid mode: {true.mode}")

def evaluate_sentiment(answer, label):
    return f1_score_str(answer, label)

def evaluate_relation_extraction(answer, label):
    return f1_score_str(answer, label)

def evaluate_headline(answer, label):
    return f1_score_str(answer, label)

from pydantic import BaseModel
class Assessment(BaseModel):
    success: bool

import litellm
def evaluate_fiqa(answer, label, task):
    system_prompt="Given the question and the ground truth, evaluate if the response answers the question."
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": "You're given the following inputs:\n\nQuestion: " + task + "\n\nGround Truth: " + label + "\n\nResponse: " + answer}]
    response = litellm.completion('gpt-4o-mini', messages=messages, temperature=0, response_format=Assessment)
    assessment = json.loads(response.choices[0].message.content)
    return int(assessment['success'])


from cognify.hub.search import default

search_settings = default.create_search(
    search_type='light',
    n_trials=10,
    opt_log_dir='mixed_opt_2',
    evaluator_batch_size=100,
)
