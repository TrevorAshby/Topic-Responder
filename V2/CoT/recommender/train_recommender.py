import torch
import datetime
import random
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoModelForCausalLM

def train(model, dataloader, tokenizer, num_epochs=10, on_gpu=True):
    print("Starting model training...")

    # input_examples = [
    #     "I like LeBron James.",
    #     "I watched Star Wars Episode 1: The Phantom Menace.",
    #     "I don't like Country Music.",
    #     "The beach is cool.",
    #     "Spaghetti good.",
    #     "I have not read Harry Potter and the Sorcerers\' Stone",
    #     "I play Minecraft."
    #     ]
    criteria = torch.optim.Adam(model.parameters(), lr= 1e-4)
    
    log = open('./log_recomm.txt', 'a+')
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
            if on_gpu:
                output = model(input_ids=x.squeeze(1).cuda(), labels=y.squeeze(1).cuda())
            else:
                output = model(input_ids=x.squeeze(1), labels=y.squeeze(1))
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
            # for in_str in input_examples:
            #     in_ids = tokenizer(in_str, return_tensors='pt').input_ids
            #     if on_gpu:
            #         example = model.generate(in_ids.cuda(), max_new_tokens=50)
            #     else:
            #         example = model.generate(in_ids, max_new_tokens=50)
            #     dec_out = tokenizer.decode(example[0], skip_special_tokens=True)
            #     log.write("\tInput:{}, Output:{}\n".format(in_str, dec_out))

            #log.write("Epoch:{}, EpLoss:{}, Input:\"{}\", Output:\"{}\"\n".format(epoch, eploss/len(dataloader), in_str, dec_out))
            #print("Epoch:{}, EpLoss:{}, Input:\"{}\", Output:\"{}\"".format(epoch, eploss/len(dataloader), in_str, dec_out))
            
            print("Epoch:{}, EpLoss:{}\n".format(epoch, eploss/len(dataloader)))
            # Generate another random example to print to console
            # in_str = input_examples[random.randrange(1,len(input_examples))]
            # in_ids = tokenizer(in_str, return_tensors='pt').input_ids
            # if on_gpu:
            #     example = model.generate(in_ids.cuda(), max_new_tokens=50)
            # else:
            #     example = model.generate(in_ids.cuda(), max_new_tokens=50)
            # dec_out = tokenizer.decode(example[0], skip_special_tokens=True)
            # print("Input:{}, Output:{}".format(in_str, dec_out))
    log.close()

class RecommenderDataset(Dataset):
    def __init__(self, input_file, tokenizer):
        self.tokenizer = tokenizer
        self.examples = open(input_file, 'r', encoding='utf-8').readlines()#pd.read_csv(input_file, sep='|')

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        inp, out = self.examples[idx].split("|")
        num_sugg = len(out.split(","))
        # prompt = f"Generate only {num_sugg} similar topics that could be suggested for new conversation that takes influence from but are not present in the following user profile: {inp} In the generated answer, generate the suggested topic within brackets [SUGGESTEDTOPIC]"
        prompt = f"Instruction: Generate only {num_sugg} similar topics that could be suggested for new conversation that takes influence from but are not present in the following user profile: {inp} In the generated answer, generate each of the suggested topics separated by a comma like so: TOPIC1,TOPIC2,TOPIC3,TOPIC4,etc.\nSuggested Topics:"
        #print(f"inp:{inp}, out:{out}, num_sugg:{num_sugg}")
        #instruction = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        inp = self.tokenizer(prompt, max_length=64, padding='max_length', truncation=True, return_tensors='pt').input_ids
        out = self.tokenizer(out, max_length=64, padding='max_length', truncation=True, return_tensors='pt').input_ids
        return inp, out


device_id = '0' # need to change this to 6 when I am training w/ jingyuan's GPU

if __name__ == '__main__':

    # set the device
    device = torch.device('cuda:'+device_id)

    # download the models
    rec_tokenizer = AutoTokenizer.from_pretrained("t5-large")
    rec_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
    # add pad token
    #rec_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    #rec_model.resize_token_embeddings(len(rec_tokenizer))

    # create dataloader 
    ds = RecommenderDataset('./recommender_data/generated_data_commas.txt', rec_tokenizer)
    dl = torch.utils.data.DataLoader(ds, batch_size=10, shuffle=True)

    # load models to GPU
    #rec_model = torch.nn.DataParallel(rec_model, device_ids=[0])
    rec_model.to(device)
    
    
    train(rec_model, dl, rec_tokenizer, 5, True)

    try:
       torch.save(rec_model.module.state_dict(), './model/rec_er.pt')
    except AttributeError:
       torch.save(rec_model.state_dict(), './model/rec_er.pt')