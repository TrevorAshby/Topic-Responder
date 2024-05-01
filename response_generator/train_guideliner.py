import torch
import datetime
import random
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration

# THIS FILE IS USED TO TRAIN OUR GUIDELINE GENERATION

def train(model, dataloader, tokenizer, num_epochs=10):
    print("Starting model training...")

    input_examples = [
        "A: Oh, I've read \"1984\". It\'s a classic. Love it. Have you ever read anything by J.K. Rowling? B: Yes, I love the Harry Potter series. Have you read them? A: Yes, I'm a big fan of Harry Potter. I have read all the books and seen all the movies. Do you have a favorite book in the series?|{\"high-level\": {\"topic\": \"literature\", \"if_interest\": \"yes\"}}, {\"middle-level\": {\"topic\": \"book recommendation\", \"if_interest\": \"unknow\"}}, {\"low-level\": {\"topic\": \"J.K. Rowling\", \"if_interest\": \"unknow\"}} {\"high-level\": {\"topic\": \"literature\", \"if_interest\": \"yes\"}}, {\"middle-level\": {\"topic\": \"book recommendation\", \"if_interest\": \"yes\"}, \"low-level\": {\"topic\": \"Harry Potter\", \"if_interest\": \"yes\"}}",
        "B: What do you like to do? A: I love eating food. I really like new restaurants.|{\"high-level\": {\"topic\": \"food\", \"if_interest\": \"yes\"}, \"low-level\": {\"topic\": \"new restaurant\", \"if_interest\": \"yes\"}}",
        "A: Have you read any good books lately? B: Yes, I just finished \"The Nightingale\" by Kristin Hannah. It was amazing. A: That sounds interesting. What is it about?|{\"high-level\": {\"topic\": \"books\", \"if_interest\": \"unknow\"}, \"middle-level\": {\"topic\": \"reading\", \"if_interest\": \"unknow\"}} {\"high-level\": {\"topic\": \"books\", \"if_interest\": \"yes\"}, \"middle-level\": {\"topic\": \"novel plot\", \"if_interest\": \"yes\"}}",
        "B: Oh, I love pasta. What\'s your favorite kind of pasta? A: I'm a big fan of spaghetti with meatballs.|{\"high-level\": {\"topic\": \"food\", \"if_interest\": \"yes\"}, \"middle-level\": {\"topic\": \"cuisine\", \"if_interest\": \"yes\"}, \"low-level\": {\"topic\": \"spaghetti\", \"if_interest\": \"yes\"}}",
        "A: By the way, have you ever gone hiking before? B: Yeah, I enjoy hiking. Why do you ask? A: I was thinking about going on a hike this weekend, would you be interested in joining me?|{\"high-level\": {\"topic\": \"outdoor activity\", \"if_interest\": \"yes\"}, \"middle-level\": {\"topic\": \"hiking\", \"if_interest\": \"yes\"}}",
    ]
    criteria = torch.optim.Adam(model.parameters(), lr= 0.000001)
    log = open('./train_guideliner_log.txt', 'a+')
    log.write('================ STARTING A NEW RUN == {} =================\n'.format(datetime.datetime.now()))
    
    for epoch in range(num_epochs):
        eploss = 0

        for batch in dataloader:
            x, y = batch
            output = model(input_ids=x.squeeze(1).cuda(), labels=y.squeeze(1).cuda())
            out = output.loss
            out.backward()
            criteria.step()
            eploss += out.item()

        if epoch % 1 == 0:
            log.write("Epoch:{}, EpLoss:{}\n".format(epoch, eploss/len(dataloader)))

            # Generate examples with each input in 'input_examples'
            for in_str in input_examples:
                in_ids = tokenizer(in_str, return_tensors='pt').input_ids
                example = model.generate(in_ids.cuda(), max_new_tokens=50)
                dec_out = tokenizer.decode(example[0], skip_special_tokens=True)
                log.write("\tInput:{}, Output:{}\n".format(in_str, dec_out))
            
            print("Epoch:{}, EpLoss:{}\n".format(epoch, eploss/len(dataloader)))

            # Generate another random example to print to console
            in_str = input_examples[random.randrange(1,len(input_examples))]
            in_ids = tokenizer(in_str, return_tensors='pt').input_ids
            example = model.generate(in_ids.cuda(), max_new_tokens=50)
            dec_out = tokenizer.decode(example[0], skip_special_tokens=True)
            print("Input:{}, Output:{}".format(in_str, dec_out))
    log.close()

# our guideline dataset class
class GuidelineDataset(Dataset):
    def __init__(self, guideline_file, tokenizer):
        self.tokenizer = tokenizer
        self.examples = pd.read_csv(guideline_file, sep='|')

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        history, pref, guideline = self.examples.iloc[idx]
        
        pref = tokenizer(pref, max_length=50, padding='max_length', truncation=True, return_tensors='pt').input_ids
        guideline = tokenizer(guideline, max_length=50, padding='max_length', truncation=True, return_tensors='pt').input_ids
        return pref, guideline

if __name__ == '__main__':
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # download the models
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

    # load model to GPU
    model.to(device)

    # create dataloader
    ds = GuidelineDataset('../data/guideliner_train_data.txt', tokenizer)
    dl = DataLoader(ds, batch_size=2, shuffle=True)

    train(model, dl, tokenizer, 20)

    try:
        torch.save(model.module.state_dict(), './model/guideliner_model.pt')
    except AttributeError:
        torch.save(model.state_dict(), './model/guideliner_model.pt')
    
