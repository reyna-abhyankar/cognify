import json
import os
import numpy as np

from cognify import register_data_loader

@cognify.register_data_loader
def load_data():
    def load_from_file(input_file):
        data_path = 'benchmark_data'
        # open the json file 
        data = json.load(open(f'{data_path}/{input_file}'))
        
        all_data = []
        for item in data:
            novice_instruction = item['simple_instruction']
            expert_instruction = item['expert_instruction']
            example_id = item['id'] 
            # directory_path = f'examples/IR_matplot_agent/new_opt_runs_test_data_mini_reason'
            directory_path = f'opt_runs'

            if not os.path.exists(directory_path):
                os.makedirs(directory_path, exist_ok=True)
            
            input = {
                'query': novice_instruction,
                "directory_path": directory_path,
                "example_id": example_id,
                "input_path": f'{data_path}/data/{example_id}',
            }
            label = {"ground_truth": f"benchmark_data/ground_truth/example_{example_id}.png"}
            all_data.append((input, label))
        return all_data
            
    all_train = load_from_file('train_data.json')
    test_data = load_from_file('test_data.json')
    train_indices = np.random.choice(range(len(all_train)), 40, replace=False).tolist()
    eval_indices = list(set(range(len(all_train))) - set(train_indices))
    
    train_data = [all_train[i] for i in train_indices]
    eval_data = [all_train[i] for i in eval_indices]
    return train_data, eval_data, test_data
    
