import os
import sys
import json
from euroeval import Benchmarker
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import set_hf_cache_env

def evaluate_validation():
    set_hf_cache_env()
    
    # Path to the trained model (or base model if not trained yet)
    model_path = "./checkpoints/final_model"
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Using base model Qwen/Qwen2.5-1.5B-Instruct for demonstration.")
        model_path = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"Starting evaluation on German Validation sets using model: {model_path}")
    
    # Initialize Benchmarker
    benchmarker = Benchmarker()
    
    # Benchmark on German language tasks
    # EuroEval/ScandEval typically evaluates on the validation/test sets defined in the tasks.
    results = benchmarker.benchmark(
        model=model_path,
        language="de",
        device="cuda",
        save_results=True,
        verbose=True
    )
    
    # Save custom report
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"logs/eval_validation_{timestamp}.json"
    
    with open(report_path, "w") as f:
        # results is usually a list of dicts or similar structure in EuroEval
        # We rely on the internal saving of EuroEval but also dump what we get
        # JSON serialization might fail if objects are not dicts, so we use str for safety
        json.dump(str(results), f, indent=4)
        
    print(f"Validation evaluation complete. Results saved to {report_path}")

if __name__ == "__main__":
    evaluate_validation()

