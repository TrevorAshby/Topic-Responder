import torch
import datetime
import random
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM

def train(model, dataloader, tokenizer, num_epochs=10):
    print("Starting model training...")

    input_examples = [
        "LeBron James is my favorite basketball player.",
        "I love the movie \"Star Wars Episode 1: The Phantom Menace\"",
        "I don't like Country Music.",
        "Let's go to the Beach!",
        "I like eating Spaghetti",
        "\"Harry Potter and the Sorcerers' Stone\" is a good book.",
        "I just played Minecraft."
        ]
    criteria = torch.optim.Adam(model.parameters(), lr= 1e-4)
    
    log = open('./log_cot.txt', 'a+')
    log.write('================ STARTING A NEW RUN == {} =================\n'.format(datetime.datetime.now()))
    
    for epoch in range(num_epochs):
        eploss = 0
        loop = tqdm(dl, leave=True)
        b = 0
        for batch in loop:
            model.train()
            criteria.zero_grad()
            x, y = batch
            # Forward pass
            output = model(input_ids=x.squeeze(1).cuda(), labels=y.squeeze(1).cuda())
            out = output.loss
            out.backward()
            criteria.step()
            eploss += out.item()
            loop.set_description(f'Epoch {epoch}, Batch {b}/{len(dl)}')
            loop.set_postfix(loss1=out.item())
            b += 1

        if epoch % 1 == 0:
            model.eval()
            log.write("Epoch:{}, EpLoss:{}\n".format(epoch, eploss/len(dataloader)))

            # Generate examples with each input in 'input_examples'
            for in_str in input_examples:
                inst_in = "Instruction: Generate a list of topics increasing in specificity to define the subject.\n\n"
                inst_in += "Input:[CONTEXT]{}[ENDOFDIALOGUE][QUESTION]The topics defining the input are".format(in_str)
                in_ids = tokenizer(inst_in, return_tensors='pt').input_ids
                example = model.generate(in_ids.cuda(), max_new_tokens=50)
                dec_out = tokenizer.decode(example[0], skip_special_tokens=True)
                log.write("\tInput:{}, Output:{}\n".format(inst_in, dec_out))
            #log.write("Epoch:{}, EpLoss:{}, Input:\"{}\", Output:\"{}\"\n".format(epoch, eploss/len(dataloader), in_str, dec_out))
            #print("Epoch:{}, EpLoss:{}, Input:\"{}\", Output:\"{}\"".format(epoch, eploss/len(dataloader), in_str, dec_out))
            print("Epoch:{}, EpLoss:{}\n".format(epoch, eploss/len(dataloader)))
            # Generate another random example to print to console
            in_str = input_examples[random.randrange(1,len(input_examples))]
            in_ids = tokenizer(in_str, return_tensors='pt').input_ids
            example = model.generate(in_ids.cuda(), max_new_tokens=50)
            dec_out = tokenizer.decode(example[0], skip_special_tokens=True)
            print("Input:{}, Output:{}".format(in_str, dec_out))
    log.close()

class TopicDataset(Dataset):
    def __init__(self, input_file, output_file, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = open(input_file, 'r', encoding='utf-8').readlines()#pd.read_csv(input_file, sep='|')
        self.outputs = open(output_file, 'r', encoding='utf-8').readlines()

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        inp = self.inputs[idx].strip()
        out = self.outputs[idx]
        #print(topic_inp)
        #print(inp)
        instruction = "Instruction: Generate a list of topics increasing in specificity to define the subject.\n\n"
        instruction += "Input:[CONTEXT]{}[ENDOFDIALOGUE][QUESTION]The topics defining the input are".format(inp)
        #topic_inp = self.tokenizer(topic_inp, max_length=50, padding='max_length', truncation=True, return_tensors='pt').input_ids
        inp = self.tokenizer(instruction, max_length=60, padding='max_length', truncation=True, return_tensors='pt').input_ids
        out = self.tokenizer(out, max_length=50, padding='max_length', truncation=True, return_tensors='pt').input_ids
        return inp, out


device_id = '0' # need to change this to 6 when I am training w/ jingyuan's GPU
max_length = 128
batch_size = 32
epoch_num = 50

if __name__ == '__main__':

    # set the device
    device = torch.device('cuda:'+device_id)

    # download the models
    inst_tokenizer = AutoTokenizer.from_pretrained("prakharz/DIAL-BART0")
    inst_model = AutoModelForSeq2SeqLM.from_pretrained("prakharz/DIAL-BART0")

    # create dataloader 
    ds = TopicDataset('./topic_xtract_data/tracker_input_cot_new.txt', './topic_xtract_data/tracker_output_cot.txt', inst_tokenizer)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    #inst_model = torch.nn.DataParallel(inst_model, device_ids=[0])

    # load models to GPU
    inst_model.to(device)
    
    
    train(inst_model, dl, inst_tokenizer, 3)

    #torch.save(blen_model.state_dict(), './model/blenderbot.pt')
    #torch.save(inst_model.state_dict(), './model/intructdialogue.pt')

    try:
        torch.save(inst_model.module.state_dict(), './model/topic_er3.pt')
    except AttributeError:
        torch.save(inst_model.state_dict(), './model/topic_er3.pt')