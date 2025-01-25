from random import randint, choice, seed, random
from tqdm import tqdm
import argparse
from datasets import load_dataset
import os
from datasets import Dataset
import re

PATTERN_TYPES = ['arithmetic', 'geometric', 'square']

def generate_sample():
    pattern = choice(PATTERN_TYPES)
    seq_length = randint(3, 8)  # Random length between 3-8
    missing_pos = randint(1, seq_length-2)
    
    # Generate base sequence
    if pattern == 'arithmetic':
        start = randint(-100, 100)
        diff = randint(-10, 10)
        if diff == 0: diff = 1  # Prevent zero difference
        seq = [start + i*diff for i in range(seq_length)]
        
    elif pattern == 'geometric':
        start = randint(1, 5)
        ratio = randint(2, 3)
        seq = [start * (ratio**i) for i in range(seq_length)]
        
    elif pattern == 'square':
        start = randint(1, 10)
        seq = [(start + i)**2 for i in range(seq_length)]
    
    # Add decimal noise (20% chance)
    if random() < 0.2:
        seq = [round(n + random()*0.9, 1) for n in seq]
    
    # Convert to strings with missing position
    answer = seq[missing_pos]
    seq_str = [str(n) if i != missing_pos else "_" 
              for i, n in enumerate(seq)]
    
    return {
        "pattern": pattern,
        "nums": seq_str,
        "target": answer
    }

def make_prefix(dp, template_type):
    sequence_str = " ".join(dp['nums'])
    if template_type == 'base':
        return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Complete the number sequence: {sequence_str}. Identify the pattern and find the missing number. Show your reasoning in <think> </think> tags and put the final numerical answer in <answer> </answer>, e.g. <answer>42</answer>.
Assistant: Let me analyze this step by step.
<think>"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/sequences')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--train_size', type=int, default=3000)
    parser.add_argument('--test_size', type=int, default=100)
    parser.add_argument('--template_type', type=str, default='base')
    
    args = parser.parse_args()
    
    # Generate samples
    samples = [generate_sample() for _ in range(args.num_samples)]
    
    # Convert to dataset
    dataset = Dataset.from_list(samples)
    train_dataset = dataset.select(range(args.train_size))
    test_dataset = dataset.select(range(args.train_size, args.train_size + args.test_size))
    
    # Create output directory
    os.makedirs(args.local_dir, exist_ok=True)
    
    # Save datasets
    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))
    
    # Test print first sample
    print("\nFirst training sample:")
    print("Sequence:", " ".join(samples[0]['nums']))
    print("Answer:", samples[0]['target'])
