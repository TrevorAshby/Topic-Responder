import torch
import datetime
import random
import torch.nn as nn
import pandas as pd
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

def train(model, dl, tokenizer, num_epochs=10, device='cpu'):
    print("Starting model training...")
    num_mods = len(os.listdir('./model/'))

    criteria = torch.optim.Adam(model.parameters(), lr= 1e-5)
    
    log = open('./log_cot.txt', 'a+')
    log.write('================ STARTING A NEW RUN == {} =================\n'.format(datetime.datetime.now()))
    
    for epoch in range(num_epochs):
        eploss = 0
        loop = tqdm(dl, leave=True)
        b = 0
        for batch in loop:
            model.train()
            criteria.zero_grad()
            # Forward pass
            batch.to(device)
            output = model(
                input_ids=batch.squeeze(1),
                labels=batch.squeeze(1),
            )
            #output = model(input_ids=x.squeeze(1).cuda(), labels=y.squeeze(1).cuda())
            out = output.loss
            out.backward()
            criteria.step()
            eploss += out.item()
            loop.set_description(f'Epoch {epoch}, Batch {b}/{len(dl)}')
            loop.set_postfix(loss1=out.item())
            b += 1

    #     if epoch % 1 == 0:
    #         model.eval()
    #         log.write("Epoch:{}, EpLoss:{}\n".format(epoch, eploss/len(dataloader)))

    #         # Generate examples with each input in 'input_examples'
    #         for in_str in input_examples:
    #             inst_in = "Instruction: Generate a list of topics increasing in specificity to define the subject.\n\n"
    #             inst_in += "Input:[CONTEXT]{}[ENDOFDIALOGUE][QUESTION]The topics defining the input are".format(in_str)
    #             in_ids = tokenizer(inst_in, return_tensors='pt').input_ids
    #             example = model.generate(in_ids.cuda(), max_new_tokens=50)
    #             dec_out = tokenizer.decode(example[0], skip_special_tokens=True)
    #             log.write("\tInput:{}, Output:{}\n".format(inst_in, dec_out))
    #         #log.write("Epoch:{}, EpLoss:{}, Input:\"{}\", Output:\"{}\"\n".format(epoch, eploss/len(dataloader), in_str, dec_out))
    #         #print("Epoch:{}, EpLoss:{}, Input:\"{}\", Output:\"{}\"".format(epoch, eploss/len(dataloader), in_str, dec_out))
    #         print("Epoch:{}, EpLoss:{}\n".format(epoch, eploss/len(dataloader)))
    #         # Generate another random example to print to console
    #         in_str = input_examples[random.randrange(1,len(input_examples))]
    #         in_ids = tokenizer(in_str, return_tensors='pt').input_ids
    #         example = model.generate(in_ids.cuda(), max_new_tokens=50)
    #         dec_out = tokenizer.decode(example[0], skip_special_tokens=True)
    #         print("Input:{}, Output:{}".format(in_str, dec_out))
        try:
            torch.save(inst_model.module.state_dict(), f'./model/cnds_model{num_mods}.pt')

            inst_model.module.save_pretrained('./hf_model/')
            inst_tokenizer.save_pretrained('./hf_model/')
        except AttributeError:
            torch.save(inst_model.state_dict(), f'./model/cnds_model{num_mods}.pt')

            inst_model.module.save_pretrained('./hf_model/')
            inst_tokenizer.save_pretrained('./hf_model/')
    log.close()
    

class TopicDataset(Dataset):
    def __init__(self, input_file, output_file, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.inputs = open(input_file, 'r', encoding='utf-8').readlines()#pd.read_csv(input_file, sep='|')
        self.outputs = open(output_file, 'r', encoding='utf-8').readlines()
        self.max_len = max_len

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        inp = self.inputs[idx].strip()
        out = self.outputs[idx]
        #print(topic_inp)
        #print(inp)
        instruction = "Instruction: Generate a list of topics increasing in specificity to define the subject of the conversation turn.\n"
        instruction += f"Conversation turn:\"{inp}\""
        formatted_prompt = (f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\nThe topics defining the input are: ```{out}```<|im_end|>\n")
        #topic_inp = self.tokenizer(topic_inp, max_length=50, padding='max_length', truncation=True, return_tensors='pt').input_ids
        enc = self.tokenizer(formatted_prompt, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
        return enc


device_id = '0' # need to change this to 6 when I am training w/ jingyuan's GPU
max_length = 150
batch_size = 32
epoch_num = 5

if __name__ == '__main__':

    # set the device
    device = torch.device('cuda:'+device_id)

    # download the models
    inst_tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3")
    inst_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3", torch_dtype=torch.float32)

    # create dataloader 
    ds = TopicDataset('./topic_xtract_data/tracker_input_cot_new.txt', './topic_xtract_data/tracker_output_cot.txt', inst_tokenizer, max_length)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    inst_model = torch.nn.DataParallel(inst_model, device_ids=[0])

    # load models to GPU
    inst_model.to(device)
    
    
    train(inst_model, dl, inst_tokenizer, epoch_num, device)