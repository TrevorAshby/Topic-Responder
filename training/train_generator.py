import json
import torch
import datetime
import torch.distributed as dist
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import AdamW
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

# THIS FILE IS USED TO TRAIN OUR RESPONSE GENERATOR

def train(model1, dl, tokenizer1, num_epochs=10):
    model1.train()
    print('Starting model training...')

    log = open('./train_generator_log.txt', 'a+')
    log.write('================ STARTING A NEW RUN == {} =================\n'.format(datetime.datetime.now()))

    criteria1 = AdamW(model1.parameters(), lr=1e-6)

    for epoch in range(num_epochs):
        eploss1 = 0

        # setup loop with TQDM and dataloader
        loop = tqdm(dl, leave=True)
        for batch in loop:
            bio, boo = batch #! FOR DETAIL ON THESE SEE LINE 76
            # initialize calculated gradients (from prev step)
            criteria1.zero_grad()
            # pull all tensor batches required for training

            input_ids1 = bio.to(device)
            # attention_mask = batch['attention_mask'].to(device)
            labels_ids1 = boo.to(device)
            # process
            outputs1 = model1(
                input_ids=input_ids1.squeeze(1),
                labels=labels_ids1.squeeze(1),
            )
            # extract loss
            loss1 = outputs1.loss
            # # calculate loss for every parameter that needs grad update
            loss1 = loss1.mean()
            
            loss1.backward()
            # update parameters
            criteria1.step()

            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss1=loss1.item())

            eploss1 += loss1.item()

        log.write("Epoch:{}, EpLoss1:{}\n".format(epoch, eploss1/len(dl)))



device_id = '0' # need to change this to 6 when I am training w/ jingyuan's GPU
max_length = 128
batch_size = 32
epoch_num = 50


# our dataset class
class GenDataset(torch.utils.data.Dataset):
    def __init__(self, blen_path, blen_tokenizer, max_length=50):
        self.blen_tokenizer = blen_tokenizer
        self.blen_examples = pd.read_csv(blen_path, sep='|')
        self.max_len = max_length

    def __len__(self):
        return len(self.blen_examples)

    def __getitem__(self, idx):
        blen_in, blen_out = self.blen_examples.iloc[idx]
    
        blen_in_out = self.blen_tokenizer(blen_in, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
        blen_out_out = self.blen_tokenizer(blen_out, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt').input_ids
        
        return blen_in_out, blen_out_out
        

if __name__ == '__main__':
    # set the device
    device = torch.device('cuda:'+device_id)

    # download the models
    blen_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400m-distill")
    blen_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400m-distill")

    # create dataloader
    ds = GenDataset('../data/responder_train_data.txt', blen_tokenizer, max_length)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    blen_model = torch.nn.DataParallel(blen_model, device_ids=[0])

    # load models to GPU
    blen_model.to(device)
    
    
    train(blen_model, dl, blen_tokenizer, 10)

    try:
        torch.save(blen_model.module.state_dict(), './model/responder_model.pt')
    except AttributeError:
        torch.save(blen_model.state_dict(), './model/responder_model.pt')

    
    