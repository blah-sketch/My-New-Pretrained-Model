%%writefile train.py
import os
import torch
import warnings
import torch.distributed as dist
from dataclasses import dataclass
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login, HfApi, snapshot_download
from transformers import (
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import load_dataset

# Suppress warnings
warnings.filterwarnings("ignore")

# ==========================================
# CUSTOM CALLBACK FOR GENERATION (FIXED FOR DDP)
# ==========================================
class SampleTextCallback(TrainerCallback):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prompts = [
            "The nature of the universe is",
            "In the future, artificial intelligence will",
            "Once upon a time there was a"
        ]

    def on_save(self, args, state, control, model=None, **kwargs):
        # Only generate on the main process to avoid duplicate prints
        if args.local_rank in [-1, 0]:
            print(f"\n\n🔍 --- GENERATING SAMPLES AT STEP {state.global_step} ---")
            
            # ---------------------------------------------------------
            # 🐛 BUG FIX: UNWRAP MODEL FOR DDP
            # ---------------------------------------------------------
            # In DDP, the model is wrapped in DistributedDataParallel.
            # We need to access '.module' to call .generate()
            inference_model = model.module if hasattr(model, "module") else model
            
            inference_model.eval()
            
            for prompt in self.prompts:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(args.device)
                
                with torch.no_grad():
                    outputs = inference_model.generate(
                        **inputs, 
                        max_new_tokens=50,   
                        do_sample=True,      
                        top_k=50,            
                        temperature=0.7      
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"📝 Prompt: {prompt}")
                print(f"🤖 Model:  {generated_text}\n")
            
            print("--------------------------------------------------\n")
            inference_model.train()

def main():
    # ==========================================
    # 1. CONFIGURATION
    # ==========================================
    @dataclass
    class Config:
        hub_username: str = "blah7"  
        model_name: str = "my-124m-llm"
        dataset_name: str = "HuggingFaceFW/fineweb-edu"
        dataset_config: str = "sample-10BT" 
        context_length: int = 1024
        
        # DDP SETTINGS
        batch_size: int = 8  
        grad_accumulation: int = 8 
        learning_rate: float = 6e-4
        
        total_steps: int = 50_000 
        save_steps: int = 500       
        save_limit: int = 2         

    config = Config()

    # ==========================================
    # 2. AUTHENTICATION
    # ==========================================
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # Initialize DDP if we are running with torchrun
    is_ddp = local_rank != -1
    if is_ddp:
        # We don't need to explicitly init_process_group here because 
        # Trainer will handle it, but we need it for our custom sync logic below.
        # However, it's safer to let Trainer handle init. 
        # We will use a simpler file-based check or just rely on Rank 0 downloading.
        pass

    if local_rank == 0 or local_rank == -1:
        try:
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")
            login(token=hf_token)
            print("✅ Logged in to Hugging Face Hub")
        except Exception as e:
            print(f"❌ Login failed: {e}")

    repo_id = f"{config.hub_username}/{config.model_name}"

    # ==========================================
    # 3. PREPARE DATASET
    # ==========================================
    dataset = load_dataset(
        config.dataset_name, 
        name=config.dataset_config, 
        split="train", 
        streaming=True
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.context_length,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text", "id", "dump", "url", "file_path", "language", "language_score", "token_count"]
    )

    shuffled_dataset = tokenized_dataset.shuffle(buffer_size=10_000, seed=42)

    # ==========================================
    # 4. INITIALIZE MODEL
    # ==========================================
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_ctx=config.context_length,
        n_positions=config.context_length,
        n_embd=768,
        n_layer=12,
        n_head=12,
        activation_function="gelu_new",
    )

    model = GPT2LMHeadModel(model_config)

    # ==========================================
    # 5. RESUME LOGIC (SYNCED ACROSS GPUs)
    # ==========================================
    # We use a simple trick: 
    # Rank 0 checks HF. If resume is needed, it downloads.
    # ALL ranks check if the local folder exists (because they share the filesystem).
    
    if local_rank == 0 or local_rank == -1:
        print("🔍 Checking for existing checkpoints...")
        api = HfApi()
        try:
            api.repo_info(repo_id)
            print(f"✅ Repo found: {repo_id}")
            # Download checkpoints so they exist on the disk
            snapshot_download(
                repo_id=repo_id,
                local_dir=config.model_name,
                allow_patterns=["checkpoint-*", "config.json", "*.json"],
                ignore_patterns=["*.md", ".gitattributes"]
            )
            print("✅ Checkpoints downloaded locally.")
        except Exception:
            print("ℹ️ Starting fresh (No checkpoints found).")

    # 🐛 BUG FIX: SYNCHRONIZATION
    # In a notebook/single-node DDP, the disk is shared.
    # We just need to wait for Rank 0 to finish downloading before Rank 1 checks.
    if is_ddp:
        if not torch.distributed.is_initialized():
             torch.distributed.init_process_group(backend="nccl")
        torch.distributed.barrier() # Rank 1 waits here until Rank 0 is done

    # Now both ranks check the disk
    resume_from_checkpoint = False
    if os.path.isdir(config.model_name) and len(os.listdir(config.model_name)) > 0:
        # Basic check: do we see checkpoint folders?
        if any("checkpoint-" in f for f in os.listdir(config.model_name)):
            resume_from_checkpoint = True
    
    if local_rank == 0:
        if resume_from_checkpoint:
            print("✅ Resuming from local checkpoint.")
        else:
            print("✅ Starting training from scratch.")

    # ==========================================
    # 6. TRAINER SETUP
    # ==========================================
    training_args = TrainingArguments(
        output_dir=config.model_name,
        overwrite_output_dir=False,
        max_steps=config.total_steps,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accumulation,
        learning_rate=config.learning_rate,
        weight_decay=0.01,
        warmup_steps=2_000,
        logging_steps=50,
        save_steps=config.save_steps,
        save_total_limit=config.save_limit,
        fp16=True,
        gradient_checkpointing=True, 
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
        push_to_hub=True,
        hub_model_id=repo_id,
        hub_strategy="checkpoint",
        report_to="none",
        optim="adamw_torch",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=shuffled_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SampleTextCallback(tokenizer)]
    )

    # ==========================================
    # 7. EXECUTION
    # ==========================================
    if local_rank == 0 or local_rank == -1:
        print("🚀 Starting Training with DDP...")

    if resume_from_checkpoint:
        try:
            # We explicitly pass True so Trainer finds the latest checkpoint
            trainer.train(resume_from_checkpoint=True)
        except Exception as e:
            if local_rank == 0:
                print(f"⚠️ Resume failed: {e}. Restarting...")
            trainer.train()
    else:
        trainer.train()

    if local_rank == 0 or local_rank == -1:
        trainer.save_model()
        trainer.push_to_hub()
        print("🎉 Session Complete.")

if __name__ == "__main__":
    main()
