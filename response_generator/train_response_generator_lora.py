# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset

# %%
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
# print(os.environ)

# %%
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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',torch_dtype=torch.float32)
model.to('cuda:0')

# Makes training faster but a little less accurate
model.config.pretraining_tp = 1

# setting padding instructions for tokenizer
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

# %%
data_files = {'train':'../data/lora_ft_train2_copy.csv', 
              'test':'../data/lora_ft_test2_copy.csv'}
dataset = load_dataset('csv', data_files=data_files, delimiter='\t', column_names=['dialogue', 'response', 'guideline'], header=None)

# %%
def form_func(sample):
    if 'agent_1' in sample['response']:
        p_in = 'agent_1'
        not_p_in = 'agent_2'
    else:
        p_in = 'agent_2'
        not_p_in = 'agent_1'

    mod_dial = sample['dialogue'].replace('person 1', 'agent_1')
    mod_dial = mod_dial.replace('person 2', 'agent_2')
    messages = [
        {
        "role":"system",
        "content": f"You are participating in the conversation. You are specifically {p_in}."
        },
        {
        "role": "user",
        "content": f"Generate the next conversation turn for {p_in} responding to {not_p_in} in this conversation: {mod_dial} Limit the generated response to 1-2 sentences and compliant with this guideline: {sample['guideline']}"
        },
        {
        "role": "assistant",
        "content": f"{p_in}:{sample['response']}"
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prompt

# %%
# Create the trainer
trainingArgs = TrainingArguments(
    output_dir=f'rg_output',
    num_train_epochs=6,
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

# %%
trainer.train()


