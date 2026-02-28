import json
from datasets import load_dataset
import os

def load_and_inspect_qasper():
    print("Loading QASPER dataset from HuggingFace...")
    # Loading the raw dataset
    dataset = load_dataset("allenai/qasper", split="train")
    
    print(f"Loaded {len(dataset)} papers.")
    
    # Inspecting one sample to understand the schema
    sample = dataset[0]
    print("\n--- Sample Schema ---")
    print(f"ID: {sample['id']}")
    print(f"Title: {sample['title']}")
    # The 'full_text' field contains the sections and paragraphs
    print(f"Section Keys: {sample['full_text']['section_name'][0:3]}...") 
    
    return dataset

if __name__ == "__main__":
    data = load_and_inspect_qasper()