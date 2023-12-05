import torch
import datetime
import random
import torch.nn as nn
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

inst_tokenizer = AutoTokenizer.from_pretrained("./hf_model/")
inst_model = AutoModelForCausalLM.from_pretrained("./hf_model/", torch_dtype=torch.float32)
inst_model.to('cuda:0')

inst_model.eval()

def generate(text_in, tok_in, mod_in):
    tok_text = tok_in(text_in, return_tensors='pt').to('cuda:0')
    gen_text = mod_in.generate(**tok_text)
    dec_text = tok_in.decode(gen_text[0], skip_special_tokens=True)
    return dec_text

while (1):
    inp = input("Enter your input: ")

    if inp == 'exit()':
        break
    
    instruction = "Instruction: Generate a list of topics increasing in specificity to define the subject of conversation.\n"
    instruction += f"Input:{inp}"
    formatted_prompt = (f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\nThe topics defining the input are:")

    out = generate(formatted_prompt, inst_tokenizer, inst_model)

    print(f"output: {out}")