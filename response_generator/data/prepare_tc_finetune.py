import os
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
import json
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import random
from tqdm import tqdm

def generate_cot(text_in, tok_in, mod_in):
    instruction = "Instruction: Generate a list of topics increasing in specificity to define the subject of conversation.\n"
    instruction += f"Input:{text_in}"
    formatted_prompt = (f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\nThe topics defining the input are:")
    tok_text = tok_in(formatted_prompt, return_tensors='pt').to('cuda:0')
    gen_text = mod_in.generate(**tok_text, max_new_tokens=60)
    dec_text = tok_in.decode(gen_text[0], skip_special_tokens=True)
    #print(dec_text)
    dec_text = re.search('```.*\n```', dec_text).group()[3:-4]

    return dec_text

dat = json.loads(open('../V2/evaluation/topical_chat/Topical-Chat/conversations/train.json', 'r').read())

cot_tokenizer = AutoTokenizer.from_pretrained("../V2/CoT/topic_extraction/hf_model_1b/")
cot_model = AutoModelForCausalLM.from_pretrained("../V2/CoT/topic_extraction/hf_model_1b/")
cot_model.to('cuda:0')

print(dat['t_bde29ce2-4153-4056-9eb7-f4ad710505fe']['content'])
print(len(dat['t_bde29ce2-4153-4056-9eb7-f4ad710505fe']['content']))

training_lines = []

for key in tqdm(dat):
    curr_lines = []

    for i in range(len(dat[key]['content'])-1):
        inst = dat[key]['content'][i]
        next = dat[key]['content'][i+1]
        curr_lines.append(f"{inst['agent']}:{inst['message']}")

        # generate guideline
        # grab a topic from the next target (next) message
        target_xtract = generate_cot(next['message'], cot_tokenizer, cot_model).strip().split('|')[0].replace('(', '').replace(')', '').split(',')
        # generate 2 topics from random indices in the conversation
        placeholder_xtract = '|'.join([generate_cot(curr_lines[random.randint(0, len(curr_lines)-1)].split(':')[-1], cot_tokenizer, cot_model).strip() for i in range(2)])

        if target_xtract[1] == 'yes':
            if next['agent'] == 'agent_1':
                # tpref = 'person2 likes'
                tpref = 'agent_2 likes'
            else:
                # tpref = 'person1 likes'
                tpref = 'agent_1 likes'
        elif target_xtract[1] == 'no':
            if next['agent'] == 'agent_1':
                # tpref = 'person2 dislikes'
                tpref = 'agent_2 dislikes'
            else:
                # tpref = 'person1 dislikes'
                tpref = 'agent_1 dislikes'
        else:
            if next['agent'] == 'agent_1':
                # tpref = 'It is unclear if the person 1 likes or dislikes'
                tpref = 'It is unclear if the agent_1 likes or dislikes'
            else:
                # tpref = 'It is unclear if the person 2 likes or dislikes'
                tpref = 'It is unclear if the agent_2 likes or dislikes'

        topic_recs = []
        topic_recs.append(target_xtract[0])
        # print(placeholder_xtract)
        for inst in placeholder_xtract.split('|'):
            inst = inst.replace('(', '').replace(')', '').split(',')
            #for subinst in inst:
            if inst[1] == 'yes' and inst[0] not in topic_recs:
                topic_recs.append(inst[0])
            
            if len(topic_recs) == 3:
                break
        
        if len(topic_recs) < 3:
            topic_recs.append(placeholder_xtract.split('|')[0].replace('(', '').replace(')', '').split(',')[0])
        
        if len(topic_recs) < 3:
            topic_recs.append(placeholder_xtract.split('|')[0].replace('(', '').replace(')', '').split(',')[1])

        guideline = f'{tpref} {target_xtract[0]}. {next["agent"]}\'s response should fall into one of the following 3 topics: {topic_recs}.'
        # print(guideline)
        training_lines.append(f"{' '.join(curr_lines)}\t\t{next['agent']}:{next['message']}\t\t{guideline}")

    # write lines to file
    df = pd.read_csv(StringIO('\n'.join(training_lines)), sep='\t\t', header=None)
    train, test = train_test_split(df, test_size=0.2)
    #train, _ = train_test_split(train, test_size=0.9)
    #test, _ = train_test_split(test, test_size=0.9)
    test = test.dropna()
    train = train.dropna()
    test.to_csv('./lora_ft_test2.csv', sep='\t')
    train.to_csv('./lora_ft_train2.csv', sep='\t')