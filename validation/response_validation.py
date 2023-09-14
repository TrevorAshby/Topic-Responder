import torch
import time
import datetime
import re
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, T5ForConditionalGeneration

# THIS FILE IS USED TO EVALUATE THE RESPONSE GENERATION OF OUR TOPIC-RESPONDER
# THE GUIDELINE MUST BE GENERATED BEFOREHAND

def main():
    #! LOAD THE MODELS
    print('loading models...')
    # topic_tokenizer = AutoTokenizer.from_pretrained('prakharz/DIAL-BART0')
    # topic_detector = AutoModelForSeq2SeqLM.from_pretrained('TrevorAshby/topic-detector')

    # guideliner_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    # guideliner = T5ForConditionalGeneration.from_pretrained("TrevorAshby/guideliner")

    blen_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    blen_model = AutoModelForSeq2SeqLM.from_pretrained("TrevorAshby/blenderbot-400M-distill")

    comp_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    comp_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

    #! SEND MODELS TO GPU
    device_id = '0'
    device = torch.device('cuda:'+device_id)

    # topic_detector.to(device)
    # guideliner.to(device)
    blen_model.to(device)
    comp_model.to(device)

    # topic_detector.eval()
    # guideliner.eval()
    blen_model.eval()
    comp_model.eval()

    log = open('./val_out.txt', 'w')

    file = open('./validation_data.txt', 'r')
    lines = file.readlines()

    the_id = 0
    # for each line, separate based upon the '[GUIDELINE] substring'
    for line in lines:
        print('{}/{}'.format(the_id, len(lines)))
        history, guideline = line.split('[GUIDELINE]')

        # pass whole line into OUR MODEL
        blend_in_ids = blen_tokenizer([line], max_length=128, return_tensors='pt', truncation=True)
        blend_example = blen_model.generate(**blend_in_ids.to(device), max_length=60)
        blend_response = blen_tokenizer.batch_decode(blend_example, skip_special_tokens=True)[0]
        generated_response = blend_response

        # pass only conversation history to BASELINE
        comp_in_ids = comp_tokenizer([history], max_length=128, return_tensors='pt', truncation=True)
        comp_example = comp_model.generate(**comp_in_ids.to(device), max_length=60)
        comp_response = comp_tokenizer.batch_decode(comp_example, skip_special_tokens=True)[0]

        # write ID, HISTORY, OUR_OUTPUT, and BASELINE_OUTPUT line to file
        log.write('{}\t{}\t{}\t{}\n'.format(the_id, history, generated_response, comp_response))
        the_id += 1

if __name__ == '__main__':
    main()