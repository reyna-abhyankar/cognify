.. _cognify_tutorials_evaluator:

******************
Workflow Evaluator
******************

Cognify evaluates your workflow throughout its optimization iterations. To tell Cognify how you want it to be evaluated, you should define an evaluator for your workflow that returns a score (a positive numerical value, higher being better generation quality) for a workflow's output. This can usually be done by comparing the output to the ground truth provided in the training dataset.

The evaluator function signature and its implementation are both customizable. A common type of signature includes workflow input, workflow output generation, and ground truth as the function parameters as follows. But you can also define an evaluation function with other or fewer parameters, e.g., an evaluator that only needs the generation output and ground truth to measure the score. To register a function as your evaluator, simply add :code:`@cognify.register_evaluator` before it.

.. code-block:: python

   @cognify.register_evaluator
   def evaluate(workflow_input, workflow_output, ground_truth):
      # your evaluation logic here
      return score

For the math-solver example, we will use LLM-as-a-judge to be the evaluator. The implementation is based on `LangChain`.

.. code-block:: python

   import cognify

   from pydantic import BaseModel
   from langchain_openai import ChatOpenAI
   from langchain_core.prompts import ChatPromptTemplate

   # Initialize the model
   import dotenv
   dotenv.load_dotenv()
   model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

   from langchain.output_parsers import PydanticOutputParser
   class Assessment(BaseModel):
      score: int
      
   parser = PydanticOutputParser(pydantic_object=Assessment)

   @cognify.register_evaluator
   def llm_judge(workflow_input, workflow_output, ground_truth):
      evaluator_prompt = """
   You are a math problem evaluator. Your task is to grade the the answer to a math proble by assessing its correctness and completeness.

   You should not solve the problem by yourself, a standard solution will be provided. 

   Please rate the answer with a score between 0 and 10.
      """
      evaluator_template = ChatPromptTemplate.from_messages(
         [
            ("system", evaluator_prompt),
            ("human", "problem:\n{problem}\n\nstandard solution:\n{solution}\n\nanswer:\n{answer}\n\nYou response format:\n{format_instructions}\n"),
         ]
      )
      evaluator_agent = evaluator_template | model | parser
      assess = evaluator_agent.invoke(
         {
            "problem": workflow_input, 
            "answer": workflow_output, 
            "solution": ground_truth, 
            "format_instructions": parser.get_format_instructions()
         }
      )
      return assess.score

The evaluator agent uses `gpt-4o-mini` as the backbone model. It also returns a structured output, ``Assessment``, to enforce the output format since we require the evaluator to return a numerical value.

Cognify provides a few `sample evaluators <https://github.com/GenseeAI/cognify/tree/main/cognify/optimizer/evaluation>`_ to start with: F1 score, LLM-as-a-judge, exact match, and code execution for the HumanEval dataset used in our examples.