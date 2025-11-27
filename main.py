import argparse
import os
from emergent_arc.detection import cca_4connected
from emergent_arc.dsl import execute_dsl
from emergent_arc.network import SpikingPolicyNetwork
from emergent_arc.evolution import Island
from emergent_arc.memory import SubroutineLibrary, VeteranPool
from emergent_arc.inference import OnlineProgramInducer
from emergent_arc.utils import load_arc_dataset, save_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Emergent-ARC v2.0")
    parser.add_argument('--data_dir', type=str, default='./data/training', help='Path to ARC training data')
    parser.add_argument('--output_dir', type=str, default='./results', help='Path to save results')
    args = parser.parse_args()
    
    # Initialize components
    subroutines = SubroutineLibrary(max_subroutines=32)
    veterans = VeteranPool(max_size=500)
    
    # Factory for creating islands/evolvers
    def evolver_factory():
        return Island(island_id=0, population_size=100, param_size=12000) # Reduced pop for demo
    
    inducer = OnlineProgramInducer(evolver_factory, subroutines, veterans)
    
    # Load ARC tasks
    # Ensure data directory exists or use dummy data
    if not os.path.exists(args.data_dir):
        print(f"Data directory {args.data_dir} not found. Creating dummy task.")
        import numpy as np
        # Create a dummy task
        tasks = [{
            'id': 'dummy_001',
            'train': [
                (np.zeros((10, 10), dtype=int), np.zeros((10, 10), dtype=int)),
                (np.ones((10, 10), dtype=int), np.ones((10, 10), dtype=int))
            ],
            'test': [{'input': np.zeros((10, 10), dtype=int), 'output': np.zeros((10, 10), dtype=int)}]
        }]
    else:
        tasks = load_arc_dataset(args.data_dir)
    
    print(f"Loaded {len(tasks)} tasks.")
    
    # Training loop
    for i, task in enumerate(tasks):
        print(f"Processing Task {task['id']} ({i+1}/{len(tasks)})...")
        
        train_pairs = task['train']
        test_input = task['test'][0]['input']
        
        # Solve with online induction
        prediction = inducer.solve_task(train_pairs, test_input)
        
        # Check result
        target = task['test'][0]['output']
        if target is not None:
            correct = (prediction == target).all()
            print(f"Task {task['id']}: {'SOLVED' if correct else 'FAILED'}")
        else:
            print(f"Task {task['id']}: Prediction generated (no ground truth)")
            
    # Save trained system
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_checkpoint({'subroutines': subroutines, 'veterans': veterans}, os.path.join(args.output_dir, 'final_checkpoint.pkl'))
    print("Training complete. Checkpoint saved.")

if __name__ == '__main__':
    main()
