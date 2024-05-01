import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, AutoPeftModelForCausalLM
from trl import SFTTrainer
from datasets import load_dataset
import argparse


def prompt_instruction_format(sample):
    num_sugg = len(sample['topics'].split(','))
    messages = [
        {
        "role":"system",
        "content": f"Your goal is to recommend new topics of conversation based on a user\'s preferences towards topics."
        },
        {
        "role": "user",
        "content": f"Generate only {num_sugg} similar topics that could be suggested for new conversation that takes influence from but are not present in the following user profile: {sample['profile']} In the generated answer, generate each of the suggested topics separated by a comma like so: TOPIC1,TOPIC2,TOPIC3,TOPIC4,etc."
        },
        {
        "role": "assistant",
        "content": f"{sample['topics']}"
        }
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return prompt

# --------------- MAIN ---------------
if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # set flags
    # parser.add_argument("-role", "--role", help="Role to use (groomer, victim)")
    # args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    # setting padding instructions for tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    # model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2",torch_dtype=torch.float32, device_map='auto')
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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',torch_dtype=torch.float32)
    # Makes training faster but a little less accurate
    model.config.pretraining_tp = 1

    data_files = {'recommender_train':'./recommender_data/recommend_train.csv', 
                'recommender_test':'./recommender_data/recommend_test.csv'}
    dataset = load_dataset('csv', data_files=data_files, delimiter='\t', column_names=['profile', 'topics'])

    # role = args.role

    # Create the trainer
    trainingArgs = TrainingArguments(
        output_dir=f'hf_output',
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
        train_dataset=dataset[f'recommender_train'],
        eval_dataset = dataset[f'recommender_test'],
        peft_config=peft_config,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=prompt_instruction_format,
        args=trainingArgs,
    )

    trainer.train()