import re
import json
import nltk
import torch
import numpy
import pickle
import codecs
import evaluate
import argparse
import networkx as nx
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from UniEval.UniEval.utils import convert_to_json
from UniEval.UniEval.metric.evaluator import get_evaluator
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# ------ UTILITIES ------ #
def form_func(sample, token_in, with_guideline):
    if 'agent_1' in sample['response']:
        p_in = 'agent_1'
        not_p_in = 'agent_2'
    else:
        p_in = 'agent_2'
        not_p_in = 'agent_1'

    mod_dial = sample['dialogue'].replace('person 1', 'agent_1')
    mod_dial = mod_dial.replace('person 2', 'agent_2')
    messages = [
        {
        "role":"system",
        "content": f"You are participating in the conversation. You are specifically {p_in}."
        }
    ]
    
    if with_guideline:
        messages.append({
        "role": "user",
        "content": f"Generate the next conversation turn for {p_in} responding to {not_p_in} in this conversation: {mod_dial} Limit the generated response to 1-2 sentences and compliant with this guideline: {sample['guideline']}"
        })
    else:
        messages.append({
        "role": "user",
        "content": f"Generate the next conversation turn for {p_in} responding to {not_p_in} in this conversation: {mod_dial} Limit the generated response to 1-2 sentences."
        })

    prompt = token_in.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


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
            #! FIXME UPDATE THIS TO THE NEW METHOD: Pos->Neut->Neg
            if g.nodes[tpxt]['pref'] == 'positive':
                if top_pref_prof[tpxt] == 'negative':
                    g.nodes[tpxt]['pref'] = 'unknown'
            if g.nodes[tpxt]['pref'] == 'unknown':
                if top_pref_prof[tpxt] == 'positive':
                    g.nodes[tpxt]['pref'] = 'positive'
                if top_pref_prof[tpxt] == 'negative':
                    g.nodes[tpxt]['pref'] = 'negative'
            if g.nodes[tpxt]['pref'] == 'negative':
                if top_pref_prof[tpxt] == 'positive':
                    g.nodes[tpxt]['pref'] = 'positive'
            # g.nodes[tpxt]['pref'] = top_pref_prof[tpxt]
            
        # add all links between nodes in chain if not already existing only if more than 1 node
        if len(top_pref_prof) > 1 and len(prev_tpxt) >= 1:
            for pt in prev_tpxt:
                if (pt.split(',')[0], tpxt.split(',')[0]) not in g.edges:
                    g.add_edge(pt.split(',')[0], tpxt.split(',')[0])
        # prev_tpxt = tpxt
        prev_tpxt.append(tpxt)  

def generate_recommendation(text_in, tok_in, mod_in):
    tok_text = tok_in(text_in, return_tensors='pt').to('cuda:0')
    gen_text = mod_in.generate(**tok_text, max_new_tokens=1024)
    dec_text = tok_in.decode(gen_text[0], skip_special_tokens=True)
    return dec_text

def generate_response(text_in, guideline, tok_in, mod_in):
    blend_in_str = text_in + ' [GUIDELINE] ' + guideline
    blend_in_ids = tok_in([blend_in_str], max_new_tokens=512, return_tensors='pt', truncation=True)
    blend_example = mod_in.generate(**blend_in_ids, max_new_tokens=60)
    response = tok_in.batch_decode(blend_example, skip_special_tokens=True)[0]
    return response

def generate_graph_eval_file(cot_model, cot_tokenizer, read_path, write_path, num_examples=50):
    ds = json.loads(open(read_path, 'r').read())
    save_js = {}
    
    # loop through each conversation
    for i, inst in enumerate(tqdm(ds)):
        if i > num_examples:
            break

        ag1_graph = nx.Graph()
        ag2_graph = nx.Graph()
        conversation = []
        pickled_graphs = {}

        # look through the utterances of the conversation
        for j, msg in enumerate(ds[inst]['content']):
            try:
                if msg['agent'] == 'agent_1':
                    utterance = msg['message']
                    # extract the topics/preferences and add them to the graph
                    topic_xtract = generate_cot(utterance, cot_tokenizer, cot_model)
                    topic_pref_profile = CoT_to_Preference(topic_xtract.strip())
                    update_graph(topic_pref_profile, ag1_graph)
                    
                    # save the graph as a pickle
                    ag1_pickled_graph = codecs.encode(pickle.dumps(ag1_graph), "base64").decode()
                    pickled_graphs[f'{len(conversation)}'] = (ag1_pickled_graph, msg['agent'], topic_pref_profile)

                else:
                    utterance = msg['message']
                    # extract the topics/preferences and add them to the graph
                    topic_xtract = generate_cot(utterance, cot_tokenizer, cot_model)
                    topic_pref_profile = CoT_to_Preference(topic_xtract.strip())
                    update_graph(topic_pref_profile, ag2_graph)
                    
                    # save the graph as a pickle
                    ag2_pickled_graph = codecs.encode(pickle.dumps(ag2_graph), "base64").decode()
                    pickled_graphs[f'{len(conversation)}'] = (ag2_pickled_graph, msg['agent'], topic_pref_profile)

                conversation.append((msg['agent'], msg['message']))
            except:
                pickled_graphs[f'{len(conversation)}'] = ('ERROR', msg['agent'], 'NULL')     
                conversation.append((msg['agent'], msg['message']))

        save_js[inst] = {"conversation":conversation,
                         "pickled_graphs":pickled_graphs,
                         "length":len(conversation)}

        # save after completing an instance
        with open(write_path, 'w') as fp:
            json.dump(save_js, fp) 
    # final save
    with open(write_path, 'w') as fp:
        json.dump(save_js, fp)             

def run_evaluation(rouge, response, ch, target, uni_evaluator):
    # Bleu
    bl = nltk.translate.bleu_score.sentence_bleu([response], target)

    # Rouge
    rg = rouge.compute(predictions=[response], references=[[target]])

    # UniEval
    data = convert_to_json(output_list=[response], 
                       src_list=ch, context_list=[''])
    uval = uni_evaluator.evaluate(data, print_result=False)

    return bl, rg, uval




def evaluate_pipeline(recc_model, recc_tokenizer, \
                      resp_model, resp_tokenizer, \
                        b1_model, b1_tokenizer, \
                            b2_model, \
                                b3_model, b3_tokenizer, \
                                    b4_model, \
                                        read_path, write_path, \
                                            num_examples=10, windows=[3,12,20]):
    
    # load rouge & unieval
    rouge = evaluate.load('rouge')
    uni_evaluator = get_evaluator('dialogue')

    # load pre-computed graph ds
    eval_ds = json.loads(open(read_path, 'r').read())

    save_js = {}

    # go through each conversation instance
    otter_idx = 0
    for i, batch in enumerate(tqdm(eval_ds)):
        if i == num_examples:
            break
        save_list = []
        # print(inst)
        # user_in = ''.join([f' person1:{c[1]}' if i % 2 == 0 else f' person2:{c[1]}' for i, c in enumerate(inst['conversation'])])
        
        # For this instance I will need to make sure to 
            # 1. check from 0-1, 0-2, 0-3, ... until reaching WIN. After slide window
            # 2. check with windows 3, 10, 20
            # Notice: Every point should be checked in this new method
        inst_js = {}
        for win in windows:
            ch_in = [eval_ds[batch]['conversation'][0], eval_ds[batch]['conversation'][1]]

            ours_list = []
            b1_list = []
            b2_list = []
            b3_list = []
            b4_list = []
            for t in range(2, len(eval_ds[batch]['conversation'])):
                user_in = ''.join([f'person 1:{c[1]}' if c[0] == 'agent_1' else f'person 2:{c[1]}' for c in ch_in])
                real_response = eval_ds[batch]['conversation'][t][1]
                current_person_predict = eval_ds[batch]['conversation'][t][0]

                # unpickle graph
                pickled = eval_ds[batch]['pickled_graphs'][str(t-1)][0]
                
                if pickled != 'ERROR':
                    unpickled = pickle.loads(codecs.decode(pickled.encode(), "base64"))

                    focus_topic = list(eval_ds[batch]['pickled_graphs'][f'{t-1}'][2].keys())[0] #! THIS SELECTION OF FOCUS TOPIC NEEDS TO BE INVESTIGATED
                    xtract_prof = {}
                    # using the focus topic, extract all connected nodes
                    xtract_prof[focus_topic] = unpickled.nodes[focus_topic]['pref']
                    for x_nodes in unpickled.edges([focus_topic]):
                        xn = x_nodes[1]
                        xtract_prof[xn] = unpickled.nodes[xn]['pref']

                    num_sugg = 3
                    prompt = f"Instruction: Generate only {num_sugg} similar topics that could be suggested for new conversation that takes influence from but are not present in the following user profile: {xtract_prof} In the generated answer, generate each of the suggested topics separated by a comma like so: TOPIC1,TOPIC2,TOPIC3,TOPIC4,etc.\nSuggested Topics:"
                    topic_recs = generate_recommendation(prompt, recc_tokenizer, recc_model).split(',')

                    # template guideline generation
                    if xtract_prof[focus_topic] == 'positive':
                        if current_person_predict == 'agent_1':
                            # tpref = 'person2 likes'
                            tpref = 'agent_2 likes'
                        else:
                            # tpref = 'person1 likes'
                            tpref = 'agent_1 likes'
                    elif xtract_prof[focus_topic] == 'negative':
                        if current_person_predict == 'agent_1':
                            # tpref = 'person2 dislikes'
                            tpref = 'agent_2 dislikes'
                        else:
                            # tpref = 'person1 dislikes'
                            tpref = 'agent_1 dislikes'
                    else:
                        if current_person_predict == 'agent_1':
                            # tpref = 'It is unclear if the person 1 likes or dislikes'
                            tpref = 'It is unclear if the agent_1 likes or dislikes'
                        else:
                            # tpref = 'It is unclear if the person 2 likes or dislikes'
                            tpref = 'It is unclear if the agent_2 likes or dislikes'

                    if current_person_predict == 'agent_1':
                        # p_in = 'person1'
                        # not_p_in = 'person2'
                        p_in = 'agent_1'
                        not_p_in = 'agent_2'
                    else:
                        # p_in = 'person2'
                        # not_p_in = 'person1'
                        p_in = 'agent_2'
                        not_p_in = 'agent_1'

                    guideline = f'{tpref} {focus_topic}. {p_in}\'s response should fall into one of the following 3 topics: {topic_recs}.'

                    # generate a response from our pipeline
                    # llama_in = f'<s>[INST] <<SYS>>\nYou are a person participating in a conversation. You are specifically {p_in}. <</SYS>>\nGenerate the next conversation turn for {p_in} responding to {not_p_in} in this conversation: {user_in} Limit the generated response to 1-2 sentences and compliant with this guideline: {guideline} [/INST] {p_in}:'
                    llama_in = form_func({'dialogue':f'{user_in}', 'response':f'{current_person_predict}', 'guideline':f'{guideline}'}, 
                                         resp_tokenizer, True)
                    blend_in_ids = resp_tokenizer(llama_in, max_length=1024, return_tensors='pt', truncation=True).to('cuda:0')
                    blend_example = resp_model.generate(blend_in_ids.input_ids, max_new_tokens=100, temperature=0.8, top_k=50, top_p = 0.85)
                    our_response = resp_tokenizer.batch_decode(blend_example, skip_special_tokens=True)[0].split('[/INST]')[-1]
                    our_response = our_response.replace('agent_1:', '')
                    our_response = our_response.replace('agent_2:', '')

                    our_bleu, our_rouge, our_unieval = run_evaluation(rouge,our_response, user_in, real_response,uni_evaluator)
                    ours = {'generated':our_response, 
                        'bleu':our_bleu, 
                        'rouge':our_rouge,
                        'unieval':our_unieval,
                        'guideline':guideline,
                        'suggested_topics':topic_recs,
                        'focus_topic':focus_topic,
                        'pref_prof':xtract_prof,
                        'input':ch_in,
                        'user_in':user_in,
                        'target':real_response}
                    # ours = {'generated':None, 
                    #     'bleu':None, 
                    #     'rouge':None,
                    #     'unieval':None,
                    #     'guideline':None,
                    #     'suggested_topics':None,
                    #     'focus_topic':None,
                    #     'pref_prof':None,
                    #     'input':ch_in,
                    #     'user_in':user_in,
                    #     'target':real_response}
                    
                    # generate response from baseline 1
                    # llama_in = f'<s>[INST] <<SYS>>\nYou are a person participating in a conversation. You are specifically {p_in}. <</SYS>>\nGenerate the next conversation turn for {p_in} responding to {not_p_in} in this conversation: {user_in} Limit the generated response to 1-2 sentences [/INST] {p_in}:'
                    llama_in = form_func({'dialogue':f'{user_in}', 'response':f'{current_person_predict}'}, 
                                         b1_tokenizer, False)
                    blend_in_ids = b1_tokenizer(llama_in, max_length=1024, return_tensors='pt', truncation=True).to('cuda:0')

                    blend_example = b1_model.generate(blend_in_ids.input_ids, max_new_tokens=100, temperature=0.8, top_k=50, top_p = 0.85)
                    b1_response = b1_tokenizer.batch_decode(blend_example, skip_special_tokens=True)[0].split('[/INST]')[-1]
                    b1_response = b1_response.replace('agent_1:', '')
                    b1_response = b1_response.replace('agent_2:', '')

                    b1_bleu, b1_rouge, b1_unieval = run_evaluation(rouge,b1_response, user_in,real_response,uni_evaluator)

                    b1 = {'generated':b1_response, 
                        'bleu':b1_bleu, 
                        'rouge':b1_rouge,
                        'unieval':b1_unieval,
                        'input':ch_in,
                        'user_in':user_in,
                        'target':real_response}
                    # b1 = {
                    #     'generated':None,
                    #     'bleu':None,
                    #     'rouge':None,
                    #     'unieval':None,
                    #     'input':ch_in,
                    #     'target':real_response
                    # }

                    # generate response from baseline 2
                    # ott_bleu, ott_rouge, ott_unieval = run_evaluation(rouge, f'{p_in}:{otter_responses[otter_idx]}', user_in, f'person2:{otter_target[otter_idx]}', uni_evaluator)
                    # ott = {
                    #     'generated':otter_responses[otter_idx],
                    #     'src/trg':otter_source[otter_idx],
                    #     'bleu':ott_bleu,
                    #     'rouge':ott_rouge,
                    #     'unieval':ott_unieval
                    # }
                    # otter_idx += 1
                    b2 = {
                        'generated':None,
                        'src/trg':None,
                        'bleu':None,
                        'rouge':None,
                        'unieval':None,
                        'input':ch_in,
                        'target':real_response
                    }

                    # generate response from baseline 3
                    inputs = b3_tokenizer(user_in + f'{p_in}:', return_tensors="pt").to('cuda:1')
                
                    generate_ids = b3_model.generate(inputs.input_ids, max_length=500, temperature=0.8, top_k=50, top_p = 0.85)
                    b3_out = b3_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                    b3_out = b3_out.split(f"{p_in}:")[-1]
                    # run evaluation
                    b3_bleu, b3_rouge, b3_unieval = run_evaluation(rouge, b3_out, user_in, real_response, uni_evaluator)
                    b3 = {
                        'generated':f'{b3_out}',
                        'bleu':b3_bleu,
                        'rouge':b3_rouge,
                        'unieval':b3_unieval,
                        'input':ch_in,
                        'user_in':user_in,
                        'target':real_response
                    }
                    # b3 = {
                    #     'generated':None,
                    #     'bleu':None,
                    #     'rouge':None,
                    #     'unieval':None,
                    #     'input':ch_in,
                    #     'target':real_response
                    # }

                    # generate a response for baseline 4
                    # completion = b4_model.chat.completions.create(model='gpt-3.5-turbo',
                    #                             messages=[{"role": "system", "content": f"You are a person participating in a conversation. You are specifically {p_in}."},
                    #                                     {"role": "user", "content": f"Generate the next conversation turn for {p_in} responding to {not_p_in} in this conversation: {user_in} Limit the generated response to 1-5 sentences."}])
                    # b4_out = completion.choices[0].message.content
                    # # run evaluation
                    # b4_bleu, b4_rouge, b4_unieval = run_evaluation(rouge, f'{p_in}:{b4_out}', user_in, real_response, uni_evaluator)
                    # b4 = {
                    #     'generated':f'person2:{b4_out}',
                    #     'bleu':b4_bleu,
                    #     'rouge':b4_rouge,
                    #     'unieval':b4_unieval,
                    #     'input':ch_in,
                    #     'target':real_response
                    # }
                    b4 = {
                        'generated':None,
                        'bleu':None,
                        'rouge':None,
                        'unieval':None,
                        'input':ch_in,
                        'target':real_response
                    }

                    ours_list.append((len(ch_in), ours))
                    b1_list.append((len(ch_in), b1))
                    b2_list.append((len(ch_in), b2))
                    b3_list.append((len(ch_in), b3))
                    b4_list.append((len(ch_in), b4))

                    # increase CH size
                    ch_in.append(eval_ds[batch]['conversation'][t])
                    if len(ch_in) > win:
                        ch_in = ch_in[1:]
            inst_js[f'window:{win}'] = {'ours':ours_list, 'b1':b1_list, 'b2':b2_list, 'b3':b3_list, 'b4':b4_list}     
    
        # save instance of all window permutations to file
        save_js[batch] = inst_js
        with open(write_path, 'w') as fp:
            json.dump(save_js, fp)


# ------ MAIN ------ #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # setup flags
    parser.add_argument("-ds", "--dataset", help="Dataset to use (tc_test_freq, tc_valid_freq, tiage, multiwoz)")
    parser.add_argument("-td", "--todo", help="Action to complete: (gen) generate evaluation file, (eva) run the evaluation *requires evaluation file")

    args = parser.parse_args()

    if args.dataset == 'tc_test_freq':
        read_path = './topical_chat/Topical-Chat/conversations/test_freq.json'
        write_path = './eval_ds_complete_V2_NEW.json'
    elif args.dataset == 'tc_valid_freq':
        read_path = './topical_chat/Topical-Chat/conversations/valid_freq.json'
        write_path = './eval_ds_complete_valid_V2_NEW.json'
    elif args.dataset == 'tiage':
        read_path = './tiage/tc_anno_test.json'
        write_path = './eval_ds_complete_tiage_V2_NEW.json'
    elif args.dataset == 'multiwoz':
        read_path = './multiwoz/tc_anno_test.json'
        write_path = './eval_ds_complete_multiwoz_V2_NEW.json'

    output_path = f'./EVAL_FINAL_{datetime.now()}_b1-4_{args.dataset}.json'

    if args.todo == 'gen':
        cot_tokenizer = AutoTokenizer.from_pretrained("../CoT/topic_extraction/hf_model_1b/")
        cot_model = AutoModelForCausalLM.from_pretrained("../CoT/topic_extraction/hf_model_1b/")
        cot_model.to('cuda:0')
        # cot_model = None
        # cot_tokenizer = None

        print('Starting Graph Eval File Generation...')
        generate_graph_eval_file(cot_model, cot_tokenizer, read_path, write_path, 2400)
    
    elif args.todo == 'eva':
        # topic recommendation model
        recc_tokenizer = AutoTokenizer.from_pretrained("../CoT/recommender/hf_model_1b/")
        recc_model = AutoModelForSeq2SeqLM.from_pretrained("../CoT/recommender/hf_model_1b/", torch_dtype=torch.float32)
        recc_model.to('cuda:0')

        # response model / b1
        # resp_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        # resp_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token='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',torch_dtype=torch.float32)
        resp_tokenizer = AutoTokenizer.from_pretrained("../../training/rg_output/checkpoint-24825/")
        resp_model = AutoModelForCausalLM.from_pretrained("../../training/rg_output/checkpoint-24825/",torch_dtype=torch.float32)
        resp_model.to('cuda:0')
        

        # OTTers / b2
        otter_responses = open('./otter/result_ep:test_valid.txt', 'r').readlines()
        otter_target = open('./otter/target_valid.csv', 'r').readlines()
        otter_source = open('./otter/source_valid.csv', 'r').readlines()
        
        b2_model = {
            "responses": otter_responses,
            "target": otter_target,
            "source": otter_source
        }

        # Vicuna / b3
        # b3_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
        # b3_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
        b3_tokenizer = AutoTokenizer.from_pretrained("../../training/rg_output_vicuna/checkpoint-8255")
        b3_model = AutoModelForCausalLM.from_pretrained("../../training/rg_output_vicuna/checkpoint-8255")

        b3_model.to('cuda:1') #! CHANGE THIS TO :1 IF SPACE ISSUE OCCURS

        # GPT3.5 / b4
        b4_model = OpenAI(api_key=open('../../api_key.txt', 'r').read())

        # recc_model = None
        # recc_tokenizer = None
        # resp_model = None
        # resp_tokenizer = None
        # b2_model = None
        # b3_model = None
        # b3_tokenizer = None
        print('Starting Pipeline Evaluation...')
        evaluate_pipeline(recc_model, recc_tokenizer, \
                          resp_model, resp_tokenizer, \
                            b1_model=resp_model, b1_tokenizer=resp_tokenizer, \
                            b2_model=b2_model, \
                                b3_model=b3_model, b3_tokenizer=b3_tokenizer, \
                                    b4_model=b4_model, \
                                        read_path=write_path, write_path=output_path, \
                                            num_examples=2400)
   