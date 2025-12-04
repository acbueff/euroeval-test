import os
import sys
import logging
import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import concatenate_datasets

# Add project root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_config, get_tokenizer, load_euroeval_datasets, format_for_qwen
from utils.paths import set_hf_cache_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train():
    # Load configuration
    config = load_config("configs/qwen_1.5b_config.yaml")
    
    # Set environment variables for Berzelius storage
    set_hf_cache_env()
    
    model_name = config["model"]["name"]
    logger.info(f"Loading model: {model_name}")
    
    # Load Tokenizer
    tokenizer = get_tokenizer(model_name)
    
    # Load Datasets
    logger.info("Loading datasets...")
    # We load the training split of the German datasets
    raw_datasets_map = load_euroeval_datasets(language=config["data"]["language"], split="train")
    
    processed_datasets = []
    for name, ds in raw_datasets_map.items():
        logger.info(f"Processing {name}...")
        # Apply formatting
        formatted_ds = ds.map(
            lambda x: format_for_qwen(x, tokenizer),
            remove_columns=ds.column_names,
            desc=f"Formatting {name}"
        )
        processed_datasets.append(formatted_ds)
    
    if not processed_datasets:
        logger.error("No datasets loaded. Exiting.")
        return

    train_dataset = concatenate_datasets(processed_datasets)
    logger.info(f"Total training samples: {len(train_dataset)}")
    
    # Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=config["model"]["trust_remote_code"],
        use_cache=config["model"]["use_cache"],
        torch_dtype=torch.bfloat16 if config["training"]["bf16"] else torch.float16,
        device_map="auto"
    )
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=float(config["training"]["learning_rate"]),
        num_train_epochs=config["training"]["num_train_epochs"],
        logging_steps=config["training"]["logging_steps"],
        save_steps=config["training"]["save_steps"],
        bf16=config["training"]["bf16"],
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        ddp_find_unused_parameters=config["training"]["ddp_find_unused_parameters"],
        report_to=config["training"]["report_to"],
        run_name="qwen-1.5b-euroeval-de"
    )
    
    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    logger.info("Starting training...")
    trainer.train()
    
    logger.info("Saving model...")
    trainer.save_model(os.path.join(config["training"]["output_dir"], "final_model"))
    tokenizer.save_pretrained(os.path.join(config["training"]["output_dir"], "final_model"))

if __name__ == "__main__":
    train()

