import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer
from datasets import Dataset, load_dataset, load_metric

class CFG:
    EXP_NAME = "honoka_rm_0323"
    
    SEQ_LEN = 128
    MODEL_NAME = "cyberagent/open-calm-small"

    MAX_STEPS = 100
    LR = 1e-4
    
    OUTPUT_DIR = f"./{EXP_NAME}"
    
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(CFG.MODEL_NAME, num_labels=2)
model.config.pad_token_id = tokenizer.eos_token_id

# dataset = load_dataset("shunk031/JGLUE", name="MARC-ja")
dataset = load_dataset('csv', data_files='honoka_sl_train_0307.csv')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def preprocess_function(batch):
    tokenized_batch = tokenizer(batch['sentence'], truncation=True, padding='max_length', max_length=CFG.SEQ_LEN)
    tokenized_batch['labels'] = batch['label']
    return tokenized_batch

train_dataset = dataset["train"].map(preprocess_function, batched=True)
print(train_dataset)

# valid_df = pd.DataFrame(dataset["validation"])
# nega_df = valid_df.query("label==1").head(250)
# posi_df = valid_df.query("label==0").sample(len(nega_df), random_state=0)
# valid_dataset = Dataset.from_pandas(pd.concat([nega_df, posi_df])).map(preprocess_function, batched=True)
dataset = load_dataset('csv', data_files='honoka_sl_valid_0307.csv')
valid_dataset = dataset["train"].map(preprocess_function, batched=True)
print(valid_dataset)

load_accuracy = load_metric("accuracy")
load_f1 = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits[0], axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"accuracy": accuracy, "f1": f1}

training_args = TrainingArguments(
    output_dir=CFG.OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    max_steps=CFG.MAX_STEPS,
    evaluation_strategy="steps",
    eval_steps=10,
    run_name=f"{CFG.EXP_NAME}",
    report_to="wandb",
    fp16=True,
    save_total_limit=5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.model.save_pretrained(f"{CFG.OUTPUT_DIR}/final_checkpoint/")