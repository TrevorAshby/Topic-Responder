import torch
import time
import datetime
import re
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, T5ForConditionalGeneration

# THIS IS THE MAIN INFERENCE FILE, THAT LOGS BOTH THE RESPONSES OF OUR MODEL AS WELL AS THE RESPONSE OF BLENDERBOT-400M-DISTILL

def main():
    START_MSG = "Hello! I am Hokiebot. I love to have conversations. Say something to me..."
    CONV_HISTORY = [('', '')]
    MAX_CONV_TURNS = 3 # set to 0 if you just want user input passed through.


    COUT_LOGGING = True

    device_id = '0'
    device = torch.device('cuda:'+device_id)
    log = open('./main_log_compare.txt', 'a+')

    generator_used = 'BLENDERBOT'

    log.write(generator_used + ' - ' + 'PREV_TURNS:{}'.format(MAX_CONV_TURNS) + ' - ' + str(datetime.datetime.now()) + '\n')
    max_length = 512


    # load the models
    print('**** LOADING MODELS ****')
    topic_tokenizer = AutoTokenizer.from_pretrained('prakharz/DIAL-BART0')
    topic_detector = AutoModelForSeq2SeqLM.from_pretrained('TrevorAshby/topic-detector')

    guideliner_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    guideliner = T5ForConditionalGeneration.from_pretrained("TrevorAshby/guideliner")

    blen_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    blen_model = AutoModelForSeq2SeqLM.from_pretrained("TrevorAshby/blenderbot-400M-distill")

    comp_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    comp_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

    # load models to device and set to eval
    print('**** LOADING CHECKPOINTS & TO GPU****')

    topic_detector.to(device)
    guideliner.to(device)
    blen_model.to(device)
    comp_model.to(device)

    topic_detector.eval()
    guideliner.eval()
    blen_model.eval()
    comp_model.eval()

    print("========== Welcome to the Hokiebot Topic based generator =========")
    print("==========      If you wish to exit, type \'goodbye\'      =========")
    print("Hokiebot: {}".format(START_MSG))
    log.write("Hokiebot: {}\n".format(START_MSG))
    while(1):
        # take in user response
        user_response = input("You: ")
        log.write("You: {}\n".format(user_response))

        # exit if the user says goodbye
        if user_response == 'goodbye':
            break

        full_time = time.time()

        # get the topic preferences
        topic_time = time.time()
        topic_modified_in = ''
        if MAX_CONV_TURNS != 0:
            for i, turn in enumerate(CONV_HISTORY):
                if turn[1] != '':
                    topic_modified_in += 'Human: ' + turn[0] + ' '
                else:
                    topic_modified_in += 'Robot: ' + turn[0] + ' '
                if (i < len(CONV_HISTORY)-1) and (i+1 % 2 == 0):
                    topic_modified_in += '[ENDOFTURN]'
        # print(topic_modified_in)
        topic_in_str = "Instruction: Extract the topic of the last conversation turn, and determine whether the human is interested in it.\n Input: [CONTEXT] " + topic_modified_in + 'Human: ' + user_response + " [ENDOFDIALOGUE] [QUESTION] Given this conversation provided, the topic and intent is"
        user_input_ids = topic_tokenizer(topic_in_str, max_length=max_length, padding='max_length', return_tensors='pt').input_ids
        topic_pref_example = topic_detector.generate(user_input_ids.to(device), max_new_tokens=128)
        topic_pref = topic_tokenizer.decode(topic_pref_example[0], skip_special_tokens=True)
        topic_time = (time.time() - topic_time)%60

        if COUT_LOGGING:
            print("=== User Input: {} ===".format(user_response))
            print("=== Topic Pref: {} ===".format(topic_pref))

        log.write("=== User Input: {} ===\n".format(user_response))
        log.write("=== Topic Pref: {} ===\n".format(topic_pref))
        # add to the conversation history, if at max, remove 1
        if len(CONV_HISTORY) == MAX_CONV_TURNS:
            CONV_HISTORY.pop(0)
        CONV_HISTORY.append((user_response, topic_pref))

        # generate the guideline with input and topic
        guideline_time = time.time()
        
        guide_in_str = ''
        topics_combi = ''
        if MAX_CONV_TURNS != 0:
            for i, turn in enumerate(CONV_HISTORY):
                # B's
                if i+1 % 2 == 0:
                    guide_in_str += 'B: ' + turn[0]
                else:
                    guide_in_str += 'A: ' + turn[0]
                topics_combi += turn[1]
            
            topics_combi += turn[-1] + ' ' + turn[-2]
            guide_in_str += '| ' + topics_combi
        else:
            guide_in_str += 'A: {}| {}'.format(user_response, topic_pref)

        in_ids = guideliner_tokenizer(guide_in_str, max_length=max_length, padding='max_length', return_tensors='pt').input_ids
        guideline_example = guideliner.generate(in_ids.to(device), max_new_tokens=50)
        guideline = guideliner_tokenizer.decode(guideline_example[0], skip_special_tokens=True)
        guideline_time = (time.time() - guideline_time)%60

        if COUT_LOGGING:
            print("=== Guideline: {} ===".format(guideline))

        log.write("=== Guideline: {} ===\n".format(guideline))

        # using the guideline generate using one of the models
        generated_response = ''
        generator_time = time.time()
        
        blend_in_str = ''
        
        if MAX_CONV_TURNS != 0:
            for i, turn in enumerate(CONV_HISTORY):
                blend_in_str += turn[0]
                if (i < len(CONV_HISTORY)-1):
                    blend_in_str += '</s> <s>'
        else:
            blend_in_str = user_response

        blend_in_str2 = ' [GUIDELINE] ' + guideline
        blend_in_ids = blen_tokenizer([blend_in_str2], max_length=128, return_tensors='pt', truncation=True)
        blend_example = blen_model.generate(**blend_in_ids.to(device), max_length=60)
        blend_response = blen_tokenizer.batch_decode(blend_example, skip_special_tokens=True)[0]
        generated_response = blend_response

        print('BLEND IN STR: ', blend_in_str)
        comp_in_ids = comp_tokenizer([blend_in_str], max_length=128, return_tensors='pt', truncation=True)
        comp_example = comp_model.generate(**comp_in_ids.to(device), max_length=60)
        comp_response = comp_tokenizer.batch_decode(comp_example, skip_special_tokens=True)[0]

        generator_time = (time.time() - generator_time)%60
        
        # add to the conversation history, if at max, remove 1
        if len(CONV_HISTORY) == MAX_CONV_TURNS:
            CONV_HISTORY.pop(0)
        CONV_HISTORY.append((generated_response, ''))

        print('Hokiebot: {} === Blen400M: {}'.format(generated_response, comp_response))
        log.write('Hokiebot: {} === Blen400M: {}\n'.format(generated_response, comp_response))
        if COUT_LOGGING:
            print('=== TIMING [tpc:{}, gdl:{}, gen:{}, tot:{}] ==='.format(topic_time, guideline_time, generator_time, (time.time() - full_time)%60))
            print('=== CONV_HISTORY [{}] ==='.format(str(CONV_HISTORY)))
        log.write('=== TIMING [tpc:{}, gdl:{}, gen:{}, tot:{}] ===\n'.format(topic_time, guideline_time, generator_time, (time.time() - full_time)%60))
        log.write('=== CONV_HISTORY [{}] ===\n'.format(str(CONV_HISTORY)))

    print('Hokiebot: Goodbye! Thanks for chatting!')
    log.write('Hokiebot: Goodbye! Thanks for chatting!\n\n')
    log.close()
    return 0

if __name__ == '__main__':
    main()