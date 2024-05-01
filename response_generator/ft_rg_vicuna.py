import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5",torch_dtype=torch.float32)
model.to('cuda:0')

# Makes training faster but a little less accurate
model.config.pretraining_tp = 1

# setting padding instructions for tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

data_files = {'train':'../data/lora_ft_train_vicuna.csv', 
              'test':'../data/lora_ft_test_vicuna.csv'}
dataset = load_dataset('csv', data_files=data_files, delimiter='\t', column_names=['dialogue', 'response'])

def form_func(sample):
    prompt = sample['dialogue'] + '\n' + sample['response']
    prompt = prompt.replace('agent_1', 'person 1')
    prompt = prompt.replace('agent_2', 'person 2')
    return prompt

# Create the trainer
trainingArgs = TrainingArguments(
    output_dir=f'rg_output_vicuna',
    num_train_epochs=5,
    per_device_train_batch_size=1,
    save_strategy="epoch",
    learning_rate=5e-4
)

peft_config = LoraConfig(
      lora_alpha=16,
      lora_dropout=0.1,
      r=64,
      bias="none",
      task_type="CAUSAL_LM",
)


trainer = SFTTrainer(
    model=model,
    train_dataset=dataset[f'train'],
    eval_dataset = dataset[f'test'],
    peft_config=peft_config,
    tokenizer=tokenizer,
    packing=True,
    formatting_func=form_func,
    args=trainingArgs,
)

trainer.train()