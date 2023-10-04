import torch
import datetime
import random
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM

def train(model, dataloader, tokenizer, num_epochs=10):
    print("Starting model training...")

    input_examples = [
        "\"LeBron James\"",
        "\"Star Wars Episode 1: The Phantom Menace\"",
        "\"Country Music\"",
        "\"Beach\"",
        "\"Spaghetti\"",
        "\"Harry Potter and the Sorcerers' Stone\"",
        "\"Minecraft\""
        ]
    criteria = torch.optim.Adam(model.parameters(), lr= 1e-4)
    
    log = open('./log.txt', 'a+')
    log.write('================ STARTING A NEW RUN == {} =================\n'.format(datetime.datetime.now()))
    
    for epoch in range(num_epochs):
        eploss = 0

        for batch in dataloader:
            model.train()
            criteria.zero_grad()
            x, y = batch
            # Forward pass
            output = model(input_ids=x.squeeze(1).cuda(), labels=y.squeeze(1).cuda())
            out = output.loss
            out.backward()
            criteria.step()
            eploss += out.item()

        if epoch % 1 == 0:
            model.eval()
            log.write("Epoch:{}, EpLoss:{}\n".format(epoch, eploss/len(dataloader)))

            # Generate examples with each input in 'input_examples'
            for in_str in input_examples:
                in_ids = tokenizer(in_str, return_tensors='pt').input_ids
                example = model.generate(in_ids.cuda(), max_new_tokens=50)
                dec_out = tokenizer.decode(example[0], skip_special_tokens=True)
                log.write("\tInput:{}, Output:{}\n".format(in_str, dec_out))
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
    def __init__(self, guideline_file, tokenizer):
        self.tokenizer = tokenizer
        self.examples = pd.read_csv(guideline_file, sep='|')

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        topic_inp, inp = self.examples.iloc[idx]
        #print(topic_inp)
        #print(inp)
        instruction = "Instruction: Generate a list of topics increasing in specificity to define the subject.\n\n"
        instruction += "Input:[CONTEXT]{}[ENDOFDIALOGUE][QUESTION]The topics defining the input are".format(inp)
        topic_inp = self.tokenizer(topic_inp, max_length=50, padding='max_length', truncation=True, return_tensors='pt').input_ids
        inp = self.tokenizer(instruction, max_length=50, padding='max_length', truncation=True, return_tensors='pt').input_ids
        return inp, topic_inp


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
    ds = TopicDataset('./ds2.txt', inst_tokenizer)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

    #inst_model = torch.nn.DataParallel(inst_model, device_ids=[0])

    # load models to GPU
    inst_model.to(device)
    
    
    train(inst_model, dl, inst_tokenizer, 30)

    #torch.save(blen_model.state_dict(), './model/blenderbot.pt')
    #torch.save(inst_model.state_dict(), './model/intructdialogue.pt')

    try:
        torch.save(inst_model.module.state_dict(), '../model/topic_er.pt')
    except AttributeError:
        torch.save(inst_model.state_dict(), '../model/topic_er.pt')