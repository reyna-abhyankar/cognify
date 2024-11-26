import base64
import json
import logging
import os
import re
import shutil
import glob
import sys
sys.path.insert(0, sys.path[0]+"/../")
import warnings

from openai import OpenAI
from cognify.graph.base import StatePool
import dotenv

dotenv.load_dotenv()

BASE_URL='https://api.openai.com/v1'
API_KEY=os.environ["OPENAI_API_KEY"]

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_4v_evaluate(ground_truth, image, rollback):
    client = OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,)
    if not os.path.exists(f'{image}'):
        if os.path.exists(f'{rollback}'):
            base64_image1 = encode_image(f"{ground_truth}")
            base64_image2 = encode_image(f"{rollback}")
        else:
            image = '/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/benchmark_data/ground_truth/empty.png'
            base64_image1 = encode_image(f"{image}")
            base64_image2 = encode_image(f"{image}")
    else:
        base64_image1 = encode_image(f"{ground_truth}")
        base64_image2 = encode_image(f"{image}")

    response = client.chat.completions.create(
      model="gpt-4o",
      temperature=0.0,
      messages=[
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": f'''You are an excellent judge at evaluating visualization plots between a model generated plot and the ground truth. You will be giving scores on how well it matches the ground truth plot.
               
               The generated plot will be given to you as the first figure. If the first figure is blank, that means the code failed to generate a figure.
               Another plot will be given to you as the second figure, which is the desired outcome of the user query, meaning it is the ground truth for you to reference.
               Please compare the two figures head to head and rate them.
               Suppose the second figure has a score of 100, rate the first figure on a scale from 0 to 100.
               Scoring should be carried out in the following aspect:
               1. Plot correctness: 
               Compare closely between the generated plot and the ground truth, the more resemblance the generated plot has compared to the ground truth, the higher the score. The score should be proportionate to the resemblance between the two plots.
               In some rare occurrence, see if the data points are generated randomly according to the query, if so, the generated plot may not perfectly match the ground truth, but it is correct nonetheless.
               Only rate the first figure, the second figure is only for reference.
               If the first figure is blank, that means the code failed to generate a figure. Give a score of 0 on the Plot correctness.
                After scoring from the above aspect, please give a final score. The final score is preceded by the [FINAL SCORE] token.
               For example [FINAL SCORE]: 40.''',
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image2}",
              },
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image1}",
              },
            },
          ],
        }
      ],
      max_tokens=1000,
    )
    pattern = r'\[FINAL SCORE\]: (\d{1,3})'


    # Create a dictionary to store settings and their scores
    match = re.search(pattern, response.choices[0].message.content)
    if match:
        score  = int(match.group(1)) / 100
    else:
        warnings.warn("No score found!!!")
        score = 0
    return score


def mainworkflow(test_sample_id, plot_path):
    ground_truth = f"/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/benchmark_data/ground_truth/example_{test_sample_id}.png"

    image = plot_path
    image_rollback = plot_path

    plot_result = gpt_4v_evaluate(ground_truth, image, image_rollback)
    print(plot_result)
  
def vision_score(gt, state: StatePool):
  pred_path = os.path.join(state.news('workspace'), state.news('plot_file_name'))
  ground_truth = f"/mnt/ssd4/lm_compiler/examples/IR_matplot_agent/benchmark_data/ground_truth/example_{state.news('sample_id')}.png"
  if not os.path.exists(ground_truth):
    return 0
  return gpt_4v_evaluate(ground_truth, pred_path, pred_path)

if __name__ == "__main__":
    # Get the number passed as an argument
    idx = int(sys.argv[1])
    plot_path = sys.argv[2]
    mainworkflow(idx, plot_path)
