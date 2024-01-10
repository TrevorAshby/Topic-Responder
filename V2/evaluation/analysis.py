# ------ IMPORTS ----- #
import nltk
import evaluate
import json
import pickle
import codecs
import networkx as nx
from evaluate import load
import torch
import numpy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from tqdm import tqdm
import re

from UniEval.UniEval.utils import convert_to_json
from UniEval.UniEval.metric.evaluator import get_evaluator


# ------ UTILITIES ------ #
def generate_cot(text_in, tok_in, mod_in):
    instruction = "Instruction: Generate a list of topics increasing in specificity to define the subject of conversation.\n"
    instruction += f"Input:{text_in}"
    formatted_prompt = (f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\nThe topics defining the input are:")
    tok_text = tok_in(formatted_prompt, return_tensors='pt').to('cuda:0')
    gen_text = mod_in.generate(**tok_text, max_length=60)
    dec_text = tok_in.decode(gen_text[0], skip_special_tokens=True)
    #print(dec_text)
    dec_text = re.search('```.*\n```', dec_text).group()[3:-4]

    return dec_text

def CoT_to_Preference(cot):
    # (sports,yes)|(football team,yes)
    # "{\"sports\":\"positive\", \"football\":\"positive\"}"
    topics = cot.split('|')
    top_dict = {}
    for top in topics:
        top = top.replace('(', '')
        top = top.replace(')', '')
        the_top, pref = top.split(',')
        # print(pref)
        if pref == 'yes':
            pref = 'positive'
        elif pref == 'no':
            pref = 'negative'
        else:
            pref = 'unknown'
        top_dict[the_top] = pref
    return top_dict

def update_graph(top_pref_prof, g):
    prev_tpxt = []
    for tpxt in top_pref_prof:
        # add node if not in graph, else update it
        if tpxt not in g.nodes:
            g.add_node(tpxt, pref=top_pref_prof[tpxt])
        else:
            g.nodes[tpxt]['pref'] = top_pref_prof[tpxt]
            
        # add all links between nodes in chain if not already existing only if more than 1 node
        if len(top_pref_prof) > 1 and len(prev_tpxt) >= 1:
            for pt in prev_tpxt:
                if (pt.split(',')[0], tpxt.split(',')[0]) not in g.edges:
                    g.add_edge(pt.split(',')[0], tpxt.split(',')[0])
        # prev_tpxt = tpxt
        prev_tpxt.append(tpxt)  

def generate_recommendation(text_in, tok_in, mod_in):
    tok_text = tok_in(text_in, return_tensors='pt').to('cuda:0')
    gen_text = mod_in.generate(**tok_text, max_length=1024)
    dec_text = tok_in.decode(gen_text[0], skip_special_tokens=True)
    return dec_text

def generate_response(text_in, guideline, tok_in, mod_in):
    blend_in_str = text_in + ' [GUIDELINE] ' + guideline
    blend_in_ids = tok_in([blend_in_str], max_length=512, return_tensors='pt', truncation=True)
    blend_example = mod_in.generate(**blend_in_ids, max_length=60)
    response = tok_in.batch_decode(blend_example, skip_special_tokens=True)[0]
    return response

# ------ LOAD MODELS ----- # 
# our pipeline
cot_tokenizer = AutoTokenizer.from_pretrained("../CoT/topic_extraction/hf_model/")
cot_model = AutoModelForCausalLM.from_pretrained("../CoT/topic_extraction/hf_model/")

recc_tokenizer = AutoTokenizer.from_pretrained("../CoT/recommender/hf_model/")
recc_model = AutoModelForSeq2SeqLM.from_pretrained("../CoT/recommender/hf_model/", torch_dtype=torch.float32)

# resp_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# resp_model = AutoModelForSeq2SeqLM.from_pretrained("TrevorAshby/blenderbot-400M-distill")

# resp_tokenizer = AutoTokenizer.from_pretrained("../../model/resp_model/")
# resp_model = AutoModelForSeq2SeqLM.from_pretrained("../../model/resp_model/")

resp_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
resp_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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',torch_dtype=torch.float32)

# baseline 1
# b1_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
# b1_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

# baseline 2
#! this is where OTTers belongs
# otter_responses = open('./otter/result_ep:test.txt', 'r').readlines()
# otter_target = open('./otter/target.csv', 'r').readlines()
# otter_source = open('./otter/source.csv', 'r').readlines()

# otter_responses = open('./otter/result_ep:test_tiage.txt', 'r').readlines()
# otter_target = open('./otter/target_tiage.csv', 'r').readlines()
# otter_source = open('./otter/source_tiage.csv', 'r').readlines()

otter_responses = open('./otter/result_ep:test_multiwoz.txt', 'r').readlines()
otter_target = open('./otter/target_multiwoz.csv', 'r').readlines()
otter_source = open('./otter/source_multiwoz.csv', 'r').readlines()

# load models to gpu
cot_model.to('cuda:0')
recc_model.to('cuda:0')
resp_model.to('cuda:0')

# b1_model.to('cuda:0')

# ------ FUNCTIONS ------ #
def generate_graph_eval_file(cot_model, cot_tokenizer, read_path, write_path, num_examples=50):
    tcds = json.loads(open(read_path, 'r').read())
    save_js = {}
    articles = []
    for i, t in enumerate(tcds):
        try:
            if tcds[t]['article_url'] not in articles:
                if len(articles) >= num_examples:
                    break

                graph = nx.Graph()
                # agent_1 is user?
                conv_list = []
                utterance = None
                ground_truth = None
                conv_history = []
                for j, msg in enumerate(tqdm(tcds[t]['content'])):
                    is_issue = False
                    if msg['agent'] == 'agent_1':
                        utterance = msg['message']
                        # generate the graph
                        #try:
                        topic_xtract = generate_cot(utterance, cot_tokenizer, cot_model)
                        topic_pref_profile = CoT_to_Preference(topic_xtract.strip())
                        update_graph(topic_pref_profile, graph)
                        focus_topic = list(topic_pref_profile.keys())[0]
                        #except:
                        #    is_issue = True

                    else:
                        ground_truth = msg['message']
                    
                    if is_issue:
                        utterance = 'Nothing'
                        ground_truth = 'Nothing'
                        focus_topic = 'Nothing'
                        pickled = 'Nothing'
                        topic_xtract = 'Nothing'
                        temp = {'utterance':utterance,'ground_truth':ground_truth, 'topic_xtract':topic_xtract,'focus_topic':focus_topic,'graph':pickled}
                        conv_list.append(temp)

                        utterance = None
                        ground_truth = None
                    if utterance != None and ground_truth != None:
                        # make graph string
                        
                        pickled = codecs.encode(pickle.dumps(graph), "base64").decode()
                        conv_history.append(utterance) # added
                        temp = {'conv_history': conv_history.copy(),'utterance':utterance,'ground_truth':ground_truth, 'topic_xtract':topic_xtract, 'focus_topic':focus_topic,'graph':pickled}
                        conv_list.append(temp)
                        conv_history.append(ground_truth) # added
                        utterance = None
                        ground_truth = None
                        
                
                save_js[t] = conv_list
                articles.append(tcds[t]['article_url'])
            else:
                continue
        except:
            continue

    with open(write_path, 'w') as fp:
        json.dump(save_js, fp)

def run_evaluation(rouge,response, ch, target,uni_evaluator):
def run_evaluation(rouge,response, ch, target,uni_evaluator):
    conv_history = [f'{target}\n']
    # c = ch.copy()
    # c.append(target)
    # conv_history = [''.join(c)]
    
    # Bleu
    bl = nltk.translate.bleu_score.sentence_bleu([response], target)

    # Rouge
    rg = rouge.compute(predictions=[response], references=[[target]])

    # UniEval
    data = convert_to_json(output_list=[response], 
                       src_list=conv_history, context_list=[''])
    uval = uni_evaluator.evaluate(data, print_result=False)

    # ChatGPT
    #! pending implementation
    #! GENERATE CHATGPT SCORE
    # messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]
    # message = blend_response
    # if message: 
    #     prompt = f'You job is to rank on a scale of 1-5 how well utterance B responds to utterance A:
    #     A: "{user_in}"
    #     B: "{message}"'
    #     messages.append( 
    #         {"role": "user", "content": prompt}, 
    #     ) 
    #     chat = openai.ChatCompletion.create( 
    #         model="gpt-3.5-turbo", messages=messages 
    #     ) 
    # reply = chat.choices[0].message.content 
    # print(f"ChatGPT: {reply}") 
    # messages.append({"role": "assistant", "content": reply}) 

    return bl, rg, uval

def evaluate_pipeline(recc_model, recc_tokenizer, resp_model, resp_tokenizer, b1_model, b1_tokenizer, read_path, write_path, num_examples=10, use_history=False):
    # load rouge & unieval
    rouge = evaluate.load('rouge')
    uni_evaluator = get_evaluator('dialogue')

    # load pre-computed graph ds
    eval_tcds = json.loads(open(read_path, 'r').read())

    save_js = {}

    # go through each conversation instance
    otter_idx = 0
    for i, t in enumerate(eval_tcds):
        if i == num_examples:
            break
        conv_list = []
        for j, inst in enumerate(tqdm(eval_tcds[t])):
            
            if use_history:
                user_in = ''.join([f' person1:{c}' if i % 2 == 0 else f' person2:{c}' for i, c in enumerate(inst['conv_history'])])
                # user_in = user_in[:-len('</s> <s>')]
            else:
                user_in = inst['utterance']


            real_response = inst['ground_truth']
            real_response = f'person2:{real_response}'
            # unpickle graph
            pickled = inst['graph']
            focus_topic = inst['focus_topic']

            if pickled != 'Nothing':
                unpickled = pickle.loads(codecs.decode(pickled.encode(), "base64"))
                
                xtract_prof = {}
                xtract_prof[focus_topic] = unpickled.nodes[focus_topic]['pref']
                for x_nodes in unpickled.edges([focus_topic]):
                    xn = x_nodes[1]
                    xtract_prof[xn] = unpickled.nodes[xn]['pref']

                num_sugg = 3
                prompt = f"Instruction: Generate only {num_sugg} similar topics that could be suggested for new conversation that takes influence from but are not present in the following user profile: {xtract_prof} In the generated answer, generate each of the suggested topics separated by a comma like so: TOPIC1,TOPIC2,TOPIC3,TOPIC4,etc.\nSuggested Topics:"
                topic_recs = generate_recommendation(prompt, recc_tokenizer, recc_model).split(',')

                # template guideline generation
                if xtract_prof[focus_topic] == 'positive':
                    tpref = 'Person1 likes'
                elif xtract_prof[focus_topic] == 'negative':
                    tpref = 'Person1 dislikes'
                else:
                    tpref = 'It is unclear if the user likes or dislikes'

                guideline = f'{tpref} {focus_topic}. person2\'s response should fall into one of the following 3 topics: {topic_recs}.'

                # generate response from our pipeline

                #! use the next 2 lines if mistral being used
                llama_in = f'<s>[INST] <<SYS>>\nYou are a person participating in a conversation. You are specifically person2. <</SYS>>\nGenerate the next conversation turn for person2 responding to person1 in this conversation: {user_in.replace("</s> <s>", " ")} Limit the generated response to 1-2 sentences and compliant with this guideline: {guideline} [/INST] person2:'
                blend_in_ids = resp_tokenizer(llama_in, max_length=1024, return_tensors='pt', truncation=True).to('cuda:0')

                #! comment this if using mistral
                # blend_in_ids = resp_tokenizer([f'{user_in} [GUIDELINE] {guideline}'], max_length=128, return_tensors='pt', truncation=True).to('cuda:0')
                
                blend_example = resp_model.generate(**blend_in_ids)
                our_response = resp_tokenizer.batch_decode(blend_example, skip_special_tokens=True)[0].split('[/INST]')[-1]
                
                our_bleu, our_rouge, our_unieval = run_evaluation(rouge,our_response, inst['conv_history'], real_response,uni_evaluator)

                ours = {'generated':our_response, 
                        'bleu':our_bleu, 
                        'rouge':our_rouge,
                        'unieval':our_unieval,
                        'guideline':guideline,
                        'suggested_topics':topic_recs,
                        'focus_topic':focus_topic,
                        'pref_prof':xtract_prof}
                
                # generate response from baseline 1
                #! use the next 2 lines if mistral being used
                llama_in = f'<s>[INST] <<SYS>>\nYou are a person participating in a conversation. You are specifically person2. <</SYS>>\nGenerate the next conversation turn for person2 responding to person1 in this conversation: {user_in.replace("</s> <s>", " ")} Limit the generated response to 1-2 sentences [/INST] person2:'
                blend_in_ids = resp_tokenizer(llama_in, max_length=1024, return_tensors='pt', truncation=True).to('cuda:0')

                #! comment this if using mistral
                # blend_in_ids = b1_tokenizer([f'{user_in}'], max_length=128, return_tensors='pt', truncation=True).to('cuda:0')
                blend_example = b1_model.generate(**blend_in_ids)
                b1_response = b1_tokenizer.batch_decode(blend_example, skip_special_tokens=True)[0].split('[/INST]')[-1]
                
                b1_bleu, b1_rouge, b1_unieval = run_evaluation(rouge,b1_response, inst['conv_history'],real_response,uni_evaluator)

                b1 = {'generated':b1_response, 
                    'bleu':b1_bleu, 
                    'rouge':b1_rouge,
                    'unieval':b1_unieval}
                
                # generate response from baseline 2
                #! this is where OTTers belongs
                # get response, conversation history, real response
                # if len(inst['conv_history']) >= 3:
                ott_bleu, ott_rouge, ott_unieval = run_evaluation(rouge, f'person2:{otter_responses[otter_idx]}', inst['conv_history'], f'person2:{otter_target[otter_idx]}', uni_evaluator)
                ott = {
                    'generated':otter_responses[otter_idx],
                    'src/trg':otter_source[otter_idx],
                    'bleu':ott_bleu,
                    'rouge':ott_rouge,
                    'unieval':ott_unieval
                }
                otter_idx += 1
                # else:
                #     ott = {
                #         'generated':"shorter than 3",
                #         'bleu':"shorter than 3",
                #         'rouge':"shorter than 3",
                #         'unieval':"shorter than 3"
                #     }

                temp = {'user_in':user_in, 'ours':ours, 'b1(vanilla)':b1, 'b2(ott)':ott, 'target_response':real_response}
                conv_list.append(temp)
                # save [input, our output, bleu, rouge, output baselines, bleu, rouge, GPT-4 ranking]
        save_js[t] = conv_list

        with open(write_path, 'w') as fp:
            json.dump(save_js, fp)


if __name__ == '__main__':
    
    #! TOPICAL CHAT
    # read_path = './topical_chat/Topical-Chat/conversations/test_freq.json'
    # write_path = './eval_ds2_complete_5000.json'
    
    #! TIAGE
    # read_path = './tiage/tc_anno_test.json'
    # write_path = './eval_ds2_complete_tiage_5000.json'
    
    #! MULTIWOZ
    read_path = './multiwoz/tc_anno_test.json'
    write_path = './eval_ds2_complete_multiwoz_1200.json'

    print('Starting Graph Eval File Generation...')
    # generate_graph_eval_file(cot_model, cot_tokenizer, read_path, write_path, 1200)

    write_path2 = './eval_final2_01022024_complete_hist.json'
    print('Starting Pipeline Evaluation...')
    evaluate_pipeline(recc_model, recc_tokenizer, resp_model, resp_tokenizer, resp_model, resp_tokenizer, write_path, write_path2, 1200, True)
    
    #evaluate_pipeline(recc_model, recc_tokenizer, resp_model, resp_tokenizer, b1_model, b1_tokenizer, write_path, write_path2, 10, True)
