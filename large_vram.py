import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# --- Configuration for GPT-2 Large on A40 GPU ---
MODEL_NAME = "gpt2-large"
# !!! IMPORTANT: User MUST update this path to the dataset location on the cloud instance !!!
DATA_FILE = "biology_qa_finetuning.jsonl" 
OUTPUT_DIR = "./results_gpt2_large_A40"      # Model checkpoints will be saved here
LOGGING_DIR = "./logs_gpt2_large_A40_tb"     # TensorBoard logs
CACHE_DIR_ROOT = "./hf_cache_A40"            # Hugging Face cache for models/datasets

# VRAM Management & Training Hyperparameters for A40 GPU with gpt2-large
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 12  # Increased for A40 VRAM
PER_DEVICE_EVAL_BATCH_SIZE = 24   # Increased for A40 VRAM
GRADIENT_ACCUMULATION_STEPS = 4   # Effective batch size = 12 * 4 = 48
LEARNING_RATE = 5e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
FP16 = torch.cuda.is_available() # Enable mixed precision if CUDA is available (A40 supports this well)
GRADIENT_CHECKPOINTING = False   # Disable for speed; A40 should handle gpt2-large without it

# --- Hugging Face Cache Setup ---
# Ensure cache directories exist and set environment variables
hf_datasets_cache_dir = os.path.join(CACHE_DIR_ROOT, "datasets")
hf_models_cache_dir = os.path.join(CACHE_DIR_ROOT, "models")
hf_home_dir = CACHE_DIR_ROOT # General Hugging Face home

os.makedirs(hf_datasets_cache_dir, exist_ok=True)
os.makedirs(hf_models_cache_dir, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOGGING_DIR, exist_ok=True)

os.environ["HF_DATASETS_CACHE"] = hf_datasets_cache_dir
os.environ["TRANSFORMERS_CACHE"] = hf_models_cache_dir
os.environ["HF_HOME"] = hf_home_dir

def main():
    print(f"Starting fine-tuning for {MODEL_NAME} on {DATA_FILE} using an A40 configuration.")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Using FP16: {FP16}")

    # 1. Load Tokenizer
    print(f"Loading tokenizer for {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=hf_models_cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token as it was None.")

    # 2. Load and Prepare Dataset
    print(f"Loading dataset from {DATA_FILE}...")
    if not os.path.exists(DATA_FILE):
        print(f"CRITICAL ERROR: Dataset file not found at {DATA_FILE}. Please ensure the path is correct.")
        return
        
    try:
        raw_dataset = load_dataset("json", data_files=DATA_FILE, cache_dir=hf_datasets_cache_dir)
        print(f"Dataset loaded. Features: {raw_dataset['train'].features}")
        print(f"Number of examples: {len(raw_dataset['train'])}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure your JSONL file has one JSON object per line, each with a 'text' field.")
        return

    if 'text' not in raw_dataset['train'].column_names:
        print(f"Error: The dataset must contain a 'text' field. Found columns: {raw_dataset['train'].column_names}")
        return
        
    # Split dataset into training and validation sets
    print("Splitting dataset into train and validation (90/10)...")
    split_dataset = raw_dataset["train"].train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(eval_dataset)}")

    def tokenize_function(examples):
        tokenized_output = tokenizer(examples["text"], truncation=True, max_length=512) # Max length for GPT-2 is 1024, 512 is a common choice
        return tokenized_output

    print("Tokenizing datasets...")
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=eval_dataset.column_names)
    print("Tokenization complete.")

    # Data Collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3. Load Model
    print(f"Loading model {MODEL_NAME}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            cache_dir=hf_models_cache_dir,
            # use_flash_attention_2=True, # Optional: requires flash-attn installed, Ampere+ GPU (A40 is Ampere)
                                          # Might need pip install flash-attn
        )
        if GRADIENT_CHECKPOINTING: # Should be False for this config
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled.")
        else:
            print("Gradient checkpointing disabled.")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    model.config.pad_token_id = tokenizer.pad_token_id

    # 4. Training Arguments
    print("Defining training arguments...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        logging_dir=LOGGING_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",    # Re-enabled for cloud
        save_strategy="epoch",
        load_best_model_at_end=True,    # Re-enabled for cloud
        metric_for_best_model="eval_loss",# Re-enabled for cloud
        fp16=FP16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING, # Pass the variable
        logging_steps=50,               # Log every 50 steps (adjust as needed)
        save_total_limit=2,
        report_to="tensorboard",
        dataloader_num_workers=2,       # Can be >0 on Linux for faster data loading
        optim="adamw_torch_fused" if FP16 and torch.cuda.is_available() and torch.__version__ >= "2.0" else "adamw_hf", # Fused optimizer for speed if conditions met
    )

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 6. Train
    print("Starting training...")
    try:
        train_result = trainer.train()
        print("Training completed.")

        trainer.save_model() 
        print(f"Model saved to {OUTPUT_DIR}")

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("Training metrics and state saved.")

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

    # 7. Evaluation (explicitly after training)
    print("Evaluating the best model on the eval dataset...")
    try:
        eval_metrics = trainer.evaluate()
        print(f"Evaluation results for the best model: {eval_metrics}")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
    except Exception as e:
        print(f"An error occurred during final evaluation: {e}")

if __name__ == "__main__":
    main() 