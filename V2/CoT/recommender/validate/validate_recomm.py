import json
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM

# download the models
cot_tokenizer = AutoTokenizer.from_pretrained("prakharz/DIAL-BART0")
cot_model = AutoModelForSeq2SeqLM.from_pretrained("prakharz/DIAL-BART0")
cot_model.load_state_dict(torch.load('../../topic_extraction/model/topic_er3.pt'))

recommender_tokenizer = AutoTokenizer.from_pretrained("t5-large")
recommender_model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
recommender_model.load_state_dict(torch.load('../model/rec_er.pt'))
recommender_model.eval()

def generate_cot(text_in):
    tok_text = cot_tokenizer(text_in, return_tensors='pt')
    # gen_text = cot_model.generate(**tok_text, max_new_tokens=50, do_sample=True, top_k=75)
    gen_text = cot_model.generate(**tok_text, max_new_tokens=50)
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

def generate_recommendation(text_in):
    tok_text = recommender_tokenizer(text_in, return_tensors='pt')
    gen_text = recommender_model.generate(**tok_text, max_new_tokens=50)
    dec_text = recommender_tokenizer.decode(gen_text[0], skip_special_tokens=True)
    return dec_text

# Validate output / shifting (using Amazon dataset I found)
output_file = open('./out_log_ext.txt', 'w')
with open('../../topical_chat/Topical-Chat-master/conversations/train.json', 'r') as jsonfile:
    topical_chat_conversations = json.load(jsonfile)
    
    for idx in range(len(topical_chat_conversations.keys())):
        if idx == 10:
            break

        instance = topical_chat_conversations[list(topical_chat_conversations.keys())[idx]]['content']
        prev_msg = ""
        for x in instance:
            #print(x['message'], x['agent'])
            if x['agent'] == 'agent_1':
                # pass input into model
                cot_out = generate_cot(prev_msg + " " + x['message'])
                # cot_out = generate_cot(x['message'])
                cot_out = cot_out.strip()
                pref = CoT_to_Preference(cot_out)
                
                num_sugg = 3
                inp = pref
                prompt = f"Instruction: Generate only {num_sugg} similar topics that could be suggested for new conversation that takes influence from but are not present in the following user profile: {inp} In the generated answer, generate each of the suggested topics separated by a comma like so: TOPIC1,TOPIC2,TOPIC3,TOPIC4,etc.\nSuggested Topics:"
                sugg_topics = generate_recommendation(prompt)
                output_file.write(f"{x['message']}|{cot_out}|{pref}|{sugg_topics}\n\n")
                print(f"{x['message']}|{cot_out}|{pref}|{sugg_topics}")
            else:
                output_file.write(f"TARGET RESPONSE: {x['message']}\n")
                prev_msg = x['message']
                print(f"TARGET RESPONSE: {x['message']}\n")
    output_file.close()