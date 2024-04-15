import torch
import gc
import wandb
import os
import numpy as np
from tqdm.auto import tqdm

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from transformers import pipeline
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, PeftConfig, prepare_model_for_kbit_training, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from trl.core import LengthSampler, respond_to_batch

class CFG:
    EXP_NAME = "honoka_ppo_0323"
    
    SEQ_LEN = 128

    REWARD_BASELINE = -20
    
    RM_BASE = "cyberagent/open-calm-small"
    RM_OUTPUT_DIR = "honoka_rm_0323/final_checkpoint"
    SFT_OUTPUT_DIR = "honoka_0307/final_checkpoint"
    SFT_MERGED_OUTPUT_DIR = "honoka_0307_merged"
    
    OUTPUT_DIR = f"./{EXP_NAME}"
    
model = AutoModelForCausalLM.from_pretrained(
    CFG.SFT_MERGED_OUTPUT_DIR, return_dict=True, load_in_8bit=True, torch_dtype=torch.bfloat16, #device_map="balanced",
)

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

peft_config = PeftConfig.from_pretrained(CFG.SFT_OUTPUT_DIR)
ppo_tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
ppo_model.gradient_checkpointing_disable = ppo_model.pretrained_model.gradient_checkpointing_disable
ppo_model.gradient_checkpointing_enable = ppo_model.pretrained_model.gradient_checkpointing_enable
ppo_model.pretrained_model.config.use_cache = True

print_trainable_parameters(ppo_model)

rm_tokenizer = AutoTokenizer.from_pretrained(CFG.RM_BASE)
rm_tokenizer.pad_token = rm_tokenizer.eos_token

sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=CFG.RM_OUTPUT_DIR,
    device_map="auto",
    model_kwargs={"load_in_8bit": True},
    tokenizer=rm_tokenizer,
    return_token_type_ids=False,
)
sentiment_pipe.model.config.pad_token_id = rm_tokenizer.eos_token_id

sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 16,
    "truncation": True,
}

# dataset = load_dataset('csv', data_files='./serif_correct/serif_train.csv')
# dataset = dataset["train"].filter(lambda example: example["makihara"] == 1)
dataset = load_dataset('csv', data_files='question100_correct.csv')
dataset = dataset["train"]
train_df = dataset.to_pandas()
texts = train_df["serif"].to_list()
# texts = [t[:5] for t in texts]  
dset = Dataset.from_dict({"query": texts})
dset = dset.map(lambda sample: ppo_tokenizer(sample['query'], truncation=True))
dset

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

# set trainer
ppo_config = {
    'batch_size': 1,
    'learning_rate': 1e-6,
    'mini_batch_size': 1,
    'gradient_accumulation_steps': 1,
    'optimize_cuda_cache': True,
    'log_with': 'wandb',
    'tracker_kwargs': {'wandb': {'name': CFG.EXP_NAME}},
    'target_kl': 0.1,
    'init_kl_coef': 0.2,
    'adap_kl_ctrl': True,
}
config = PPOConfig(**ppo_config)
ppo_trainer = PPOTrainer(
    config,
    ppo_model,
    ref_model=None,
    tokenizer=ppo_tokenizer,
    dataset=dset,
    data_collator=collator,
)

generation_kwargs = {
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": 1,
    "eos_token_id": 0,
}

output_min_length = 8
output_max_length = 32
output_length_sampler = LengthSampler(output_min_length, output_max_length)

results = []
for ite, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
    if ite > 100:
        print("stop")
        break
    
    # generation_kwargs["min_new_tokens"] = int(min[ite])
    # generation_kwargs["max_new_tokens"] = 32
    query_tensors = [torch.tensor(i).to(f"cuda:{ppo_model.current_device}") for i in batch["input_ids"]]

    ppo_model.gradient_checkpointing_disable()
    ppo_model.pretrained_model.config.use_cache = True

    response_tensors = ppo_trainer.generate(
        query_tensors, return_prompt=False, length_sampler=output_length_sampler, **generation_kwargs
    )
    # response_tensors = ppo_trainer.generate(
    #     query_tensors, return_prompt=False, **generation_kwargs
    # )
    # print(response_tensors)

    if torch.sum(response_tensors[0]) == 0:
        print(ite)
        continue

    batch["response"] = ppo_tokenizer.batch_decode(response_tensors)

    # Compute sentiment score
    texts = [q + r for q, r in zip(batch["query"], batch["response"])]
    print('----texts----')
    print(texts)

    pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
    
    rewards = [torch.tensor(output[0]["score"] - CFG.REWARD_BASELINE) for output in pipe_outputs]
    print('----reward----')
    print(rewards)

    ppo_model.gradient_checkpointing_enable()
    ppo_model.pretrained_model.config.use_cache = False
    
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    stats['ppo/policy/ratio'] = [0]
    print(stats['objective/kl'])
    if stats['objective/kl'] > 50 or stats['objective/kl'] < -50:
        print("stop")
        break
    ppo_trainer.log_stats(stats, batch, rewards)

    # ppo_trainer.save_pretrained(f"{CFG.OUTPUT_DIR}/checkpoint-{ite}")
    
    if ite > 1 and (ite + 1) % 20 == 0:
        ppo_trainer.save_pretrained(f"{CFG.OUTPUT_DIR}/checkpoint-{ite}")
        
    gc.collect()
    torch.cuda.empty_cache()
        
ppo_trainer.save_pretrained(f"{CFG.OUTPUT_DIR}/final_checkpoint")

wandb.finish()