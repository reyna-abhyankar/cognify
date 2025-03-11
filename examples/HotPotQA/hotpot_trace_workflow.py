import os
import dotenv
dotenv.load_dotenv()

import dspy

gpt4o = dspy.LM('gpt-4o', max_tokens=1000)
colbert = dspy.ColBERTv2(url=os.environ['COLBERT_URL'])

dspy.configure(lm=gpt4o, rm=colbert)

from dsp.utils.utils import deduplicate
from trace_config import *

@model
class BasicMultiHopQA():
  def __init__(self, passages_per_hop):
    self.retrieve = dspy.Retrieve(k=passages_per_hop)
  
  def forward(self, question):
    context = []

    search_query = generate_query(question).data.search_query
    passages = self.retrieve(search_query).passages
    context = deduplicate(context + passages)
    
    search_query = generate_query_with_context(context, question).data.search_query
    passages = self.retrieve(search_query).passages
    context = deduplicate(context + passages)

    answer = generate_answer(context, question).data.answer
    return answer
    
agent = BasicMultiHopQA(passages_per_hop=2)

all_train, all_val, all_test = load_all_data()

import autogen
from opto.trace import node, bundle, model, ExecutionError
from opto.trace.nodes import GRAPH, ParameterNode
from opto.optimizers import OptoPrime
from tqdm import tqdm

## train
epochs = 5
total_train_acc = 0

optimizer = OptoPrime(agent.parameters(), config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))

for i in range(epochs):
    for j, example in enumerate(tqdm(all_train)):
        GRAPH.clear()
        try:
            response = agent(question=example.question)
            try:
                correctness = answer_f1(example.answer, response.data)
                if correctness > 0.8:
                    feedback = "The answer is correct! No need to change anything."
                else:
                    feedback = f"The answer is wrong. We expect the output of your answer to be \"{example.answer}\". Please modify the prompt and relevant parts of the program to help LLM produce the right answer."
                no_error = True
                total_train_acc += correctness
            except:
                correctness = 0
                no_error = False
        except ExecutionError as e:
            response = e.exception_node
            feedback = response.data
            correctness = 0
            no_error = False

        optimizer.zero_feedback()
        optimizer.backward(response, feedback)

        print(f"output={response.data}, feedback={feedback}\n")  # logging

        optimizer.step(verbose=False)
        checkpoint_name = f"trace_results/epoch_{i}/{j}.pkl"
        agent.save(checkpoint_name)

## evaluation
def eval(test_set):
    total_score = 0
    for example in tqdm(test_set):
        try:
            result = agent(question=example.question)
            score = answer_f1(example.answer, result.data)
            total_score += score
        except:
            pass
    return total_score / len(test_set)

val_acc = eval(all_test)