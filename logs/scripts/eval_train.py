import os
import sys
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import load_config, load_euroeval_datasets, format_for_qwen
from utils.paths import set_hf_cache_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_train():
    """
    Evaluates the model on the training split of the datasets.
    Calculates Perplexity as a proxy for fit.
    """
    set_hf_cache_env()
    config = load_config("configs/qwen_1.5b_config.yaml")
    
    model_path = "./checkpoints/final_model"
    if not os.path.exists(model_path):
        logger.warning(f"Trained model not found at {model_path}. Using base model.")
        model_path = config["model"]["name"]

    logger.info(f"Loading model from {model_path} for training split evaluation...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        device_map="auto",
        torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16
    )
    model.eval()

    # Load training datasets
    datasets_map = load_euroeval_datasets(language=config["data"]["language"], split="train")
    
    results = {}
    
    for name, ds in datasets_map.items():
        logger.info(f"Evaluating on {name} (train split)...")
        
        # Take a subset for speed if needed
        subset = ds.select(range(min(len(ds), 1000))) 
        
        nlls = []
        
        for example in tqdm(subset, desc=f"Perplexity {name}"):
            formatted = format_for_qwen(example, tokenizer)
            text = formatted["text"]
            
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                neg_log_likelihood = outputs.loss
                
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).mean())
        results[name] = {"perplexity": ppl.item()}
        logger.info(f"{name} Perplexity: {ppl.item()}")

    # Save results
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_train_results.json", "w") as f:
        import json
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    evaluate_train()

