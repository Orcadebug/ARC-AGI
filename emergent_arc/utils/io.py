import json
import pickle
import os
import numpy as np
from typing import List, Dict

def load_arc_dataset(path: str) -> List[Dict]:
    """
    Loads ARC dataset from JSON files.
    Returns a list of task dictionaries.
    """
    tasks = []
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist.")
        return []
        
    for filename in os.listdir(path):
        if filename.endswith('.json'):
            with open(os.path.join(path, filename), 'r') as f:
                task_data = json.load(f)
                
                # Convert lists to numpy arrays
                processed_task = {'id': filename.split('.')[0], 'train': [], 'test': []}
                
                for pair in task_data['train']:
                    processed_task['train'].append((
                        np.array(pair['input']),
                        np.array(pair['output'])
                    ))
                    
                for pair in task_data['test']:
                    processed_task['test'].append({
                        'input': np.array(pair['input']),
                        'output': np.array(pair['output']) if 'output' in pair else None
                    })
                    
                tasks.append(processed_task)
    return tasks

def save_checkpoint(obj: object, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_checkpoint(path: str) -> object:
    with open(path, 'rb') as f:
        return pickle.load(f)
