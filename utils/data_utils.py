import os
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from .paths import set_hf_cache_env

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_euroeval_datasets(language="de", split="train"):
    """
    Loads relevant German datasets often used in EuroEval benchmarks.
    Note: EuroEval/ScandEval is an evaluation framework. 
    This function loads the underlying raw datasets for training/evaluation purposes.
    """
    # Ensure cache is set
    set_hf_cache_env()
    
    datasets_list = []
    
    # Example German datasets commonly used
    # Note: This is illustrative. The actual EuroEval benchmark downloads specific versions.
    # Ideally, we load from the cache directly or use the same HF IDs.
    
    dataset_ids = [
        "deepset/germanquad", # QA
        "germeval_14",        # NER (requires parsing) -> often simplified to text classification for LLMs or use specific NER template
        "xnli"                # NLI (German subset)
    ]

    loaded_data = {}
    
    for ds_id in dataset_ids:
        try:
            if ds_id == "xnli":
                ds = load_dataset(ds_id, language, split=split)
            else:
                ds = load_dataset(ds_id, split=split)
            loaded_data[ds_id] = ds
            print(f"Successfully loaded {ds_id} ({split})")
        except Exception as e:
            print(f"Error loading {ds_id}: {e}")
            
    return loaded_data

def format_for_qwen(example, tokenizer):
    """
    Format example for Qwen Instruct fine-tuning.
    This assumes a generic 'text' and 'label' or 'question'/'answer' format.
    Needs customization based on specific dataset columns.
    """
    # Placeholder formatting logic
    instruction = ""
    input_text = ""
    output_text = ""

    # Heuristic for different dataset types
    if "question" in example and "answers" in example: # QA
        instruction = "Answer the following question in German."
        input_text = example["question"]
        # answers is usually a dict
        output_text = example["answers"]["text"][0] if example["answers"]["text"] else ""
        
    elif "premise" in example and "hypothesis" in example: # NLI
        instruction = "Determine if the hypothesis is entailed by the premise."
        input_text = f"Premise: {example['premise']}\nHypothesis: {example['hypothesis']}"
        output_text = str(example["label"])
        
    else:
        # Generic fallback
        instruction = "Process the following text."
        input_text = str(example.get("text", ""))
        output_text = str(example.get("label", ""))

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        {"role": "assistant", "content": output_text}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}

