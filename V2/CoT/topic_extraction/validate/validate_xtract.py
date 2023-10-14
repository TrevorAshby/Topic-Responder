import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM

# download the models
cot_tokenizer = AutoTokenizer.from_pretrained("prakharz/DIAL-BART0")
cot_model = AutoModelForSeq2SeqLM.from_pretrained("prakharz/DIAL-BART0")
cot_model.load_state_dict(torch.load('../model/topic_er3.pt'))

def generate_cot(text_in):
    tok_text = cot_tokenizer(text_in, return_tensors='pt')
    gen_text = cot_model.generate(**tok_text)
    dec_text = cot_tokenizer.decode(gen_text[0], skip_special_tokens=True)
    return dec_text

def CoT_to_Preference(cot):
    # (sports,yes)|(football team,yes)
    # "{\"sports\":\"positive\", \"football\":\"positive\"}"
    print(cot)
    topics = cot.split('|')
    top_dict = {}
    for top in topics:
        top = top.replace('(', '')
        top = top.replace(')', '')
        the_top, pref = top.split(',')
        #print(pref)
        if pref == 'yes':
            pref = 'positive'
        elif pref == 'no':
            pref = 'negative'
        else:
            pref = 'unknown'
        top_dict[the_top] = pref
    return top_dict

# Validate output / shifting (using Amazon dataset I found)
output_file = open('./out_log_ext.txt', 'w')
with open('../../topical_chat/Topical-Chat-master/conversations/train.json', 'r') as jsonfile:
    topical_chat_conversations = json.load(jsonfile)
    
    for idx in range(len(topical_chat_conversations.keys())):
        if idx == 20:
            break

        instance = topical_chat_conversations[list(topical_chat_conversations.keys())[idx]]['content']
        for x in instance:
            #print(x['message'], x['agent'])
            if x['agent'] == 'agent_2':
                # pass input into model
                cot_out = generate_cot(x['message'])
                cot_out = cot_out.strip()
                pref = CoT_to_Preference(cot_out)
                
                output_file.write(f"{x['message']}|{cot_out}|{pref}\n")
                print(f"{x['message']}|{cot_out}|{pref}")
            else:
                output_file.write(f"TARGET RESPONSE: {x['message']}\n")
                print(f"TARGET RESPONSE: {x['message']}\n")
    output_file.close()