import os
import sys
import json
from euroeval import Benchmarker
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import set_hf_cache_env

def evaluate_test():
    set_hf_cache_env()
    
    model_path = "./checkpoints/final_model"
    if not os.path.exists(model_path):
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"Starting evaluation on German Test sets using model: {model_path}")
    
    # Initialize Benchmarker
    # Note: By default EuroEval evaluates on the 'test' split if available and public, 
    # otherwise 'validation'. 
    benchmarker = Benchmarker()
    
    results = benchmarker.benchmark(
        model=model_path,
        language="de",
        device="cuda",
        save_results=True
    )
    
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"logs/eval_test_{timestamp}.json"
    
    with open(report_path, "w") as f:
        json.dump(str(results), f, indent=4)
        
    print(f"Test evaluation complete. Results saved to {report_path}")

if __name__ == "__main__":
    evaluate_test()

