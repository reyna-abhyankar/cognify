import re
from collections import defaultdict

import dspy.teleprompt
from prompts import leader_system_message, role_system_message
import json
from pydantic import BaseModel
import dotenv
import dspy
from dspy.teleprompt import MIPROv2
from dspy.evaluate import Evaluate
from dspy_agents import *
from config import load_data, evaluate_trace
import time
from tqdm import tqdm
from opto.trace import node, bundle, model, ExecutionError
from opto.trace.nodes import GRAPH
from opto.optimizers import OptoPrime
import autogen

dotenv.load_dotenv()

@bundle(trainable=True, allow_external_dependencies=True)
def format_history(history: list[str]):
    if not history:
        return "Empty"
    hist_str = ""
    for i, hist in enumerate(history):
        hist_str += f"\n---- Step {i+1} ----\n{hist}"
    # print(f"formatted history: {hist_str}")
    return hist_str

def construct_fin_agent(agent_name):
    if agent_name == "Value_Factor_Researcher":
        return dspy.Predict(ValueFactorResearcher)
    elif agent_name == "Growth_Factor_Researcher":
        return dspy.Predict(GrowthFactorResearcher)
    elif agent_name == "Momentum_Factor_Researcher":
        return dspy.Predict(MomentumFactorResearcher)
    elif agent_name == "Quality_Factor_Researcher":
        return dspy.Predict(QualityFactorResearcher)
    elif agent_name == "Volatility_Factor_Researcher":
        return dspy.Predict(VolatilityFactorResearcher)
    elif agent_name == "Liquidity_Factor_Researcher":
        return dspy.Predict(LiquidityFactorResearcher)
    elif agent_name == "Sentiment_Factor_Researcher":
        return dspy.Predict(SentimentFactorResearcher)
    elif agent_name == "Macro_Factor_Researcher":
        return dspy.Predict(MacroFactorResearcher)
    elif agent_name == "Portfolio_Manager":
        return dspy.Predict(PortfolioManager)
    elif agent_name == "Quantitative_Analyst":
        return dspy.Predict(QuantitativeAnalyst)
    elif agent_name == "Financial_Data_Specialist":
        return dspy.Predict(FinancialDataSpecialist)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

@model
class FinAgent():
    def __init__(self, agent_name):
        self.fin_robot = construct_fin_agent(agent_name)
        self.task_history = defaultdict(list)
    
    @bundle(trainable=True, allow_external_dependencies=True)
    def solve_order(self, order, task):
        hist = format_history(self.task_history[task]).data
        response = self.fin_robot(
            history=hist, current_order=order
        ).response
        self.task_history[task].append(
            f"Order: {order}\n"
            f"My Response: {response}"
        )
        return response
    
@bundle(trainable=True, allow_external_dependencies=True)
def parse_order_string(order_string: str):
    pattern = r"\[(.*?)\]\s+(.*)"
    match = re.search(pattern, order_string)
    
    if match:
        name = match.group(1)  # Extract name inside square brackets
        order = match.group(2)  # Extract the order instruction
        return name, order
    else:
        raise ValueError(f"Invalid order string format: {order_string}. Ensure it follows '[<name of staff>] <order>'.")

with open('agent_profiles.json') as f:
    profiles = json.load(f)
group_members: dict[str, FinAgent] = {}
for profile in profiles:
    agent_name = profile['name']
    agent_role = profile['profile']
    group_members[agent_name] = FinAgent(agent_name)

class GroupLeader(dspy.Signature):
    """
    You are the leader of the following group members:
    
    Member information in format \"- <member_name>: <member_role>\"
    
    - Value_Factor_Researcher: As a value factor researcher, the individual must possess expertise in financial statement analysis, a strong understanding of valuation metrics, adeptness in Python for quantitative modeling, and the ability to work collaboratively in team settings to integrate the value perspective into broader investment strategies.
    - Growth_Factor_Researcher: As a growth factor researcher, the individual must possess expertise in analyzing corporate growth indicators like earnings and revenue expansion, have strong Python skills for data analysis, and collaborate effectively in group settings to evaluate investment growth opportunities.
    - Momentum_Factor_Researcher: As a momentum factor researcher, one needs to have the ability to identify and analyze market trends and price patterns, be proficient in Python for statistical analysis, and work collaboratively in a team to leverage momentum-based investment strategies.
    - Quality_Factor_Researcher: As a quality factor researcher, the individual should evaluate companies based on financial health and earnings quality, utilize Python for quantitative analysis, and engage in team discussions to integrate quality assessments into investment decisions.
    - Volatility_Factor_Researcher: As a volatility factor researcher, one must analyze price fluctuations and risk metrics, demonstrate strong Python skills for risk modeling, and contribute to team efforts in developing risk-adjusted trading strategies.
    - Liquidity_Factor_Researcher: As a liquidity factor researcher, the position requires the ability to assess asset tradeability and market depth, use Python for liquidity analysis, and collaborate with the team to incorporate liquidity insights into trading algorithms.
    - Sentiment_Factor_Researcher: As a sentiment factor researcher, the individual should analyze market sentiment and investor opinions, be adept in Python for processing and analyzing large sentiment data sets, and work with colleagues to factor sentiment analysis into market predictions.
    - Macro_Factor_Researcher: As a macro factor researcher, one needs to understand the impact of macroeconomic indicators on markets, have strong Python skills for econometric analysis, and engage collaboratively in aligning investment strategies with macroeconomic conditions.
    - Portfolio_Manager: As a portfolio manager, the individual must integrate findings from various factor analyses to create and manage comprehensive investment strategies, demonstrate proficiency in Python for strategy development, and work collaboratively to ensure that these strategies meet the firm’s investment goals and risk tolerance.
    - Quantitative_Analyst: As a quantitative analyst, one is responsible for validating investment strategies and factors, conducting back-tests and risk assessments using Python, and collaborating with the team to ensure that the investment approach is both statistically sound and aligned with risk management protocols.
    - Financial_Data_Specialist: As a financial information officer, the individual is responsible for gathering, processing, analyzing, and extracting key financial information from structured and unstructured data sources.
    
    As a group leader, you are responsible for coordinating the team's efforts to complete a project. You will be given a user task, history progress and the remaining number of orders you can make. Please try to complete the task without exceeding the order limit.
    
    Your role is as follows:
    - Summarize the status of the project progess.
    - Based on the progress, you can decide whether to make a new order or to end the project. 
        * If you believe the task is completed, set the project status to "END" and give the final solution based on the conversation history.
    - If you need to give an order to one of your team members to make further progress:
        * Orders should follow the format: \"[<name of staff>] <order>\". 
            - The name of the staff must be wrapped in square brackets, followed by the order after a space.
        * Ensure that each order is clear, detailed, and actionable.
        * If a group member is seeking clarification/help, provide additional information to help them complete the task or make an order to another member to collect the necessary information.
    - Only issue one order at a time.
    """
    task: str = dspy.InputField()
    project_history: str = dspy.InputField()
    remaining_order_budget: int = dspy.InputField()
    project_status: str = dspy.OutputField()
    member_order: str = dspy.OutputField()
    solution: str = dspy.OutputField()

@model
class FinRobot():
    def __init__(self, k=5):
        self.k = k
        self.task_history = defaultdict(list)
        self.group_leader = dspy.Predict(GroupLeader)

    @bundle(trainable=True, allow_external_dependencies=True)
    def forward(self, task):
        for i in range(self.k):
            project_hist = format_history(self.task_history[task])
            leader_msg = self.group_leader(
                task=task, project_history=project_hist, remaining_order_budget=self.k - i
            )
            
            if leader_msg.project_status == "END":
                return leader_msg
            
            try:
                result = parse_order_string(leader_msg.member_order).data
                member_name, member_order = result
                if member_name in group_members:
                    member_response = group_members[member_name].solve_order(member_order, task)
                    
                else:
                    member_response = "Invalid group member was chosen."
                self.task_history[task].append(
                    f"Order: {member_name} - {member_order}\n"
                    f"Member Response: {member_response}"
                )
            except ValueError as e:
                self.task_history[task].append(
                    f"Error in parsing the member order"
                )
        else:
            return dspy.Prediction(solution="Project not completed in time.")

        
fin_robot = FinRobot(k=3)

lm = dspy.LM("gpt-4o-mini", temperature=0, cache=False)
dspy.settings.configure(lm=lm)

## load data ##
all_train, _, all_test = load_data()

# convert to examples
def convert_to_example(input):
    return dspy.Example({"task": input[0]["task"], "mode": input[0]["mode"], "label": input[1]["label"]}).with_inputs("task")

all_train_examples = [convert_to_example(input) for input in all_train]
all_test_examples = [convert_to_example(input) for input in all_test]

## train
epochs = 1
total_train_acc = 0

optimizer = OptoPrime(fin_robot.parameters(), config_list=autogen.config_list_from_json("OAI_CONFIG_LIST"))

for i in range(epochs):
    for j, example in enumerate(tqdm(all_train_examples)):
        GRAPH.clear()
        try:
            response = fin_robot(task=example.task)
            try:
                correctness = evaluate_trace(example, response.data)
                if correctness > 0.8:
                    feedback = "The answer is correct! No need to change anything."
                else:
                    feedback = f"The answer is wrong. We expect the output of your answer to be \"{example.label}\". Please modify the prompt and relevant parts of the program to help LLM produce the right answer."
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
        fin_robot.save(checkpoint_name)

## evaluation
def eval(test_set):
    total_score = 0
    for example in tqdm(test_set):
        try:
            result = fin_robot(task=example.task)
            score = evaluate_trace(example, result.data)
            total_score += score
        except:
            pass
    return total_score / len(test_set)

val_acc = eval(all_test_examples)
