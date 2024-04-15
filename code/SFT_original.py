from tqdm.auto import tqdm

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset

class CFG:
    EXP_NAME = "honoka_0307"
    
    SEQ_LEN = 128
    MODEL_NAME = "cyberagent/open-calm-3b"

    TRUE_BS = 8
    BS = 4
    GRAD_ACCUM = TRUE_BS // BS
    MAX_STEPS = 100
    LR = 1e-4
    
    OUTPUT_DIR = f"./{EXP_NAME}"

dataset = load_dataset('csv', data_files='honoka_merged.csv')
# positive_train = dataset["train"].filter(lambda example: example["oda"] == 1)
# dataset = load_dataset('csv', data_files='serif_correct/serif_valid.csv')
# positive_valid = dataset["train"].filter(lambda example: example["oda"] == 1)
positive_train = dataset["train"].filter(lambda example: example["label"] == 0)
positive_valid = dataset["train"].filter(lambda example: example["label"] == 1)
print(positive_train)
print(positive_valid)

def prepare_sample_text(example):
    return example["sentence"]
    # return example["serif"]

def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

chars_per_token = chars_token_ratio(positive_train, tokenizer)

train_dataset = ConstantLengthDataset(
        tokenizer,
        positive_train,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=CFG.SEQ_LEN,
        chars_per_token=chars_per_token,
)
valid_dataset = ConstantLengthDataset(
        tokenizer,
        positive_valid,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=CFG.SEQ_LEN,
        chars_per_token=chars_per_token,
)

train_dataset.start_iteration = 0

lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
        per_device_train_batch_size=CFG.BS, 
        per_device_eval_batch_size=CFG.BS,
        gradient_accumulation_steps=CFG.GRAD_ACCUM,
        eval_accumulation_steps=CFG.GRAD_ACCUM,
        max_steps=CFG.MAX_STEPS,
        learning_rate=CFG.LR,
        save_steps=10,
        fp16=True,
        logging_steps=10,
        save_total_limit=1,
        output_dir=CFG.OUTPUT_DIR,
        optim="paged_adamw_8bit",
        evaluation_strategy='steps',
        run_name=f"{CFG.EXP_NAME}",
        report_to="wandb",
)

model = AutoModelForCausalLM.from_pretrained(
    CFG.MODEL_NAME, load_in_8bit=True, device_map="auto"
)
model.config.use_cache = False 

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    peft_config=lora_config,
    packing=True,
)

trainer.model.print_trainable_parameters()

trainer.train()

trainer.model.save_pretrained(f"{CFG.OUTPUT_DIR}/final_checkpoint/")