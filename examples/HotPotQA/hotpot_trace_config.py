from cognify.hub.evaluators import f1_score_str
from cognify.hub.datasets.hotpot import HotPotQA
import litellm
from pydantic import BaseModel
import json

def answer_f1(answer: str, ground_truth: str):
    return f1_score_str(answer, ground_truth) # score function

def load_data():
    dataset = HotPotQA(train_seed=1, 
                       train_size=150, 
                       eval_seed=2023, 
                       dev_size=200, 
                       test_size=0)
    return dataset.train[0:100], dataset.train[100:150], dataset.dev # training and evaluation data

from opto.trace import bundle

class QueryOutput(BaseModel):
    search_query: str

@bundle(trainable=True, allow_external_dependencies=True)   
def generate_query(question):
    """Call an LLM to generate a search query based on the given question."""

    messages = [
        {"role": "system", "content": "You are an expert at crafting precise search queries based on a provided question. Your task is to generate a well-structured search query that will help retrieve relevant external documents containing information needed to answer the question."},
        {"role": "user", "content": f"Given the following question, please generate a search query that will help retrieve relevant external documents containing information needed to answer the question.\n\nQuestion: {question}"},
    ]
    response = litellm.completion("gpt-4o", messages, response_format=QueryOutput)
    return QueryOutput(**json.loads(response.choices[0].message.content))

@bundle(trainable=True, allow_external_dependencies=True)   
def generate_query_with_context(context, question):
    """Call an LLM to generate a search query based on the given context and question."""

    messages = [
        {"role": "system", "content": "You are good at extracting relevant details from the provided context and question. Your task is to propose an effective search query that will help retrieve additional information to answer the question. The search query should target the missing information while avoiding redundancy."},
        {"role": "user", "content": f"Given the following context and question, please generate a search query that will help retrieve relevant external documents containing information needed to answer the question.\n\nQuestion: {context} \n\nQuestion: {question}"},
    ]
    response = litellm.completion("gpt-4o", messages, response_format=QueryOutput)
    return QueryOutput(**json.loads(response.choices[0].message.content))


class AnswerOutput(BaseModel):
    answer: str

@bundle(trainable=True, allow_external_dependencies=True)   
def generate_answer(context, question):
    """Call an LLM to generate an answer to the question based on the given context."""

    messages = [
        {"role": "system", "content": "You are an expert at answering questions based on provided documents. Your task is to formulate a clear, accurate, and concise answer to the given question by using the retrieved context (documents) as your source of information. Please ensure that your answer is well-grounded in the context and directly addresses the question."},
        {"role": "user", "content": f"Given the following context and question, please generate a search query that will help retrieve relevant external documents containing information needed to answer the question.\n\nQuestion: {context} \n\nQuestion: {question}"},
    ]
    response = litellm.completion("gpt-4o", messages, response_format=AnswerOutput)
    return AnswerOutput(**json.loads(response.choices[0].message.content))
