# !pip install unsloth
#!pip install xformers

# If above doesn't work, try:
# !pip install --no-deps bitsandbytes accelerate xformers==0.0.29 peft trl triton
# !pip install --no-deps cut_cross_entropy unsloth_zoo
# !pip install sentencepiece protobuf datasets huggingface_hub hf_transfer
# !pip install --no-deps unsloth

import sys
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(REPO_ROOT))
import utils.utils as utils
import utils.prompts as prompts
from importlib import reload
from finetuning_utils import (
    REPO_ROOT,
    build_finetuning_datasets,
    build_run_name,
    run_finetuned_evaluation,
)

reload(utils)
reload(prompts)

from unsloth import FastLanguageModel
import torch
import re 
from tqdm import tqdm
from datasets import load_dataset
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import numpy as np
import pandas as pd
from datasets import Dataset
import argparse
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datetime import datetime
import json

max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.


def main():
    parser = argparse.ArgumentParser(description="Finetuning script for medical LLMs.")
    parser.add_argument('--cpt', action='store_true', help='Enable Continuous Pre-Training.')
    parser.add_argument('--para', type=int, default=1, choices=[1, 5, 10], help='Level of paraphrasing for CPT handbook. 0 for original, 5 for 5x, 10 for 10x.')
    parser.add_argument('--dataset', type=str, required=True, choices=['handbook', 'ktas', 'mimic'], help='Dataset to use for finetuning.')
    parser.add_argument('--model_name', type=str, default="unsloth/Qwen2.5-7B", help='Name of the model to use for finetuning.')
    parser.add_argument('--device_batch_size', type=int, default=2, help='Batch size per device during training.')
    parser.add_argument('--peft', action='store_true', help='Enable PEFT for finetuning.')
    parser.add_argument('--output_dir', type=str, default=str(REPO_ROOT / "results" / "finetuned_models"), help='Directory to save finetuned artifacts.')
    parser.add_argument('--skip_eval', action='store_true', help='Skip immediate post-training evaluation.')
    args = parser.parse_args()

    assert args.device_batch_size % 16 == 0 and args.device_batch_size <= 16, "Batch size must be a divisor of 8 and not larger than 8."

    if args.peft:
        cpt_learning_rate = 4e-5
        finetuning_learning_rate = 2e-4
    else:
        cpt_learning_rate = 15e-6
        finetuning_learning_rate = 3e-5

    if args.dataset == 'handbook':
        num_train_epochs = 20
    elif args.dataset == 'ktas':
        num_train_epochs = 20
    else: # mimic
        num_train_epochs = 10

    cpt_num_train_epochs = 0
    if args.cpt:
        if args.para > 0:
            cpt_num_train_epochs = int(30 / args.para)
        else:
            cpt_num_train_epochs = 30

    filename = build_run_name(
        model_name=args.model_name,
        dataset=args.dataset,
        num_train_epochs=num_train_epochs,
        finetuning_learning_rate=finetuning_learning_rate,
        cpt_num_train_epochs=cpt_num_train_epochs,
        para=args.para,
        peft=args.peft,
    )
    timestamp = datetime.now().strftime("%m-%d_%H-%M")
    filename_with_timestamp = f"{filename}_{timestamp}"
    artifact_dir = Path(args.output_dir) / filename_with_timestamp

    max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
    dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = 1024,   # Context length - can be longer, but uses more memory
        #load_in_4bit = load_in_4bit,     # 4bit uses much less memory
        load_in_8bit = False,    # A bit more accurate, uses 2x memory
        full_finetuning = not args.peft, # We have full finetuning now!
        # token = "hf_...",      # use one if using gated models
    )

    if args.peft:
        model = FastLanguageModel.get_peft_model(
            model,
            r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 32,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )

    EOS_TOKEN = tokenizer.eos_token


    if args.cpt:
        #  --- DATA for CPT -----

        # Load the cleaned text file
        if args.para == 1:
            handbook_path = REPO_ROOT / "data/ESI-Handbook/cleaned_handbook.txt"
        else:
            handbook_path = REPO_ROOT / f"data/ESI-Handbook/paragraphed_{args.para}x_handbook.txt"

        with open(handbook_path, "r", encoding="utf-8") as f:
            text = f.read()

        # Further clean the text
        def remove_newline_between_chars(text):
            return re.sub(r"(\S)\n(\S)", r"\1 \2", text)

        # Example usage
        cleaned_text = remove_newline_between_chars(text)

        # Tokenize and chunk text
        def chunk_text(text, max_length):
            """Splits text into chunks while maintaining sentence boundaries."""
            import re
            sentences = re.split(r"(?<=[.!?]) +", text)  # Split by sentence
            chunks, current_chunk = [], ""
            
            for sentence in sentences:
                if len(tokenizer.encode(current_chunk + sentence)) < max_length:
                    current_chunk += " " + sentence
                else:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence

            if current_chunk:
                chunks.append(current_chunk.strip())

            return chunks

        # Create chunks
        chunks = chunk_text(cleaned_text, max_seq_length)

        # Add EOS token at the end of each chunk
        
        formatted_chunks = [{"text": chunk} for chunk in chunks]

        # Convert to Hugging Face dataset
        handbook = Dataset.from_list(formatted_chunks)

        ## --- CPT ---- 

        if args.para > 0:
            cpt_num_train_epochs = int(30 / args.para)
        else:
            cpt_num_train_epochs = 30

        trainer = UnslothTrainer(
            model = model,
            tokenizer = tokenizer,
            train_dataset = handbook,
            dataset_text_field = "text",
            max_seq_length = max_seq_length,
            dataset_num_proc = 1,
            args = UnslothTrainingArguments(
                run_name = f"CPT_{filename_with_timestamp}",
                per_device_train_batch_size = args.device_batch_size,
                gradient_accumulation_steps = int(16/int(args.device_batch_size)),

                warmup_steps=5,
                num_train_epochs = cpt_num_train_epochs,

                learning_rate = cpt_learning_rate,
                # embedding_learning_rate = 5e-6,
                save_strategy = "no",
                fp16 = not is_bfloat16_supported(),
                bf16 = is_bfloat16_supported(),
                logging_steps = 1,
                optim = "adamw_8bit",
                weight_decay = 0.01,
                lr_scheduler_type = "cosine",
                seed = 3407,
                report_to = "wandb", # Use this for WandB etc
            ),
        )

        trainer_stats = trainer.train()
        # print(trainer_stats)

    ###  ----Finetuning ---- 
    _, _, dataset, test_dataset, true_acuity_col = build_finetuning_datasets(dataset=args.dataset, eos_token=EOS_TOKEN)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 1,
        packing = True, # Can make training 5x faster for short sequences.
        args = TrainingArguments(
            run_name=f"Finetune_{filename_with_timestamp}",
            per_device_train_batch_size = args.device_batch_size,
            gradient_accumulation_steps = int(16/args.device_batch_size),
            num_train_epochs = num_train_epochs, # Set this for 1 full training run.
            learning_rate = finetuning_learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            save_strategy = "no",
            weight_decay = 0.01,
            warmup_ratio=0.1,
            lr_scheduler_type = "cosine",
            seed = 3407,
            report_to = "wandb", # Use this for WandB etc
        ),
    )

    trainer_stats = trainer.train()
    # print(trainer_stats)

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(artifact_dir))
    tokenizer.save_pretrained(str(artifact_dir))
    metadata = {
        "base_model_name": args.model_name,
        "dataset": args.dataset,
        "peft": args.peft,
        "cpt": args.cpt,
        "para": args.para,
        "filename": filename,
        "filename_with_timestamp": filename_with_timestamp,
        "artifact_dir": str(artifact_dir),
    }
    (artifact_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved finetuned artifacts to {artifact_dir}")

    if args.skip_eval:
        print("Skipping immediate evaluation.")
        return

    FastLanguageModel.for_inference(model)
    output_dir_name = f"Triage-{args.dataset.upper() if args.dataset != 'handbook' else 'Handbook'}"
    output_prefix = REPO_ROOT / "results" / output_dir_name / filename_with_timestamp
    metrics, _ = run_finetuned_evaluation(
        model=model,
        tokenizer=tokenizer,
        test_dataset=test_dataset,
        true_acuity_col=true_acuity_col,
        output_prefix=str(output_prefix),
    )

    print("Overall Metrics:", metrics)
    print(f"\nTriage Analysis:")
    print(f"  Overtriage:  {metrics['overall']['overtriage_count']} ({metrics['overall']['overtriage_rate']:.2%})")
    print(f"  Undertriage: {metrics['overall']['undertriage_count']} ({metrics['overall']['undertriage_rate']:.2%})")
    print(f"  Off by ≥2:   {metrics['overall']['outside_by_2_count']} ({metrics['overall']['outside_by_2_rate']:.2%})")
    print("Evaluation complete. Predictions and metrics saved.")

if __name__ == '__main__':
    main()
