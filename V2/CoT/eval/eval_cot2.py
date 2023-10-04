import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig,\
      T5ForConditionalGeneration, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
# from scipy.special import softmax

# load the models
# CoT
cot_tokenizer = AutoTokenizer.from_pretrained("prakharz/DIAL-BART0")
cot_model = AutoModelForSeq2SeqLM.from_pretrained("prakharz/DIAL-BART0")
cot_model.load_state_dict(torch.load('../../model/topic_er2.pt'))

cot_model.cuda()
cot_model.eval()

# sent_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
# sent_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
# config = AutoConfig.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# sent_model.cuda()

# inst_tokenizer = AutoTokenizer.from_pretrained("prakharz/DIAL-BART0")
# inst_model = AutoModelForSeq2SeqLM.from_pretrained("prakharz/DIAL-BART0")

# inst_model.cuda()

# original extractor
topic_tokenizer = AutoTokenizer.from_pretrained('prakharz/DIAL-BART0')
topic_detector = AutoModelForSeq2SeqLM.from_pretrained('TrevorAshby/topic-detector')

topic_detector.cuda()

input_sentences = open('../../../data/generated_data/train/0_0_master_train_clean.txt', 'r', encoding='utf-8').readlines()
out_log = open('./out_log2.txt', 'w')
for i, sent in enumerate(input_sentences):
    if i == 50:break

    s_in = sent.split('|')[0].split(':')[-1]
    print(s_in)

    # input into original topic extract
    user_response = s_in

    topic_in_str = "Instruction: Extract the topic of the last conversation turn, and determine whether the human is interested in it.\n Input: [CONTEXT] " + 'Human: ' + user_response + " [ENDOFDIALOGUE] [QUESTION] Given this conversation provided, the topic and intent is"
    user_input_ids = topic_tokenizer(topic_in_str, max_length=250, padding='max_length', return_tensors='pt').input_ids
    topic_pref_example = topic_detector.generate(user_input_ids.cuda(), max_new_tokens=128)
    topic_pref = topic_tokenizer.decode(topic_pref_example[0], skip_special_tokens=True)

    # input into new topic extract
    # def extract_topic_sentiment(text_in):
    #     instruct_input = "Instruction:What is the topic of conversation?\n\nInput:[CONTEXT]{}[ENDOFDIALOGUE][QUESTION]The topic of conversation is".format(text_in)
    #     tokens_input = inst_tokenizer(instruct_input, max_length=250, padding='max_length', truncation=True, return_tensors='pt').to("cuda:0")
    #     input_out = inst_model.generate(**tokens_input)
    #     topic = inst_tokenizer.decode(input_out[0], skip_special_tokens=True)

    #     tokens_input = sent_tokenizer(text_in, max_length=250, padding='max_length', truncation=True, return_tensors='pt').to("cuda:0")
    #     input_out = sent_model(**tokens_input)

    #     scores = softmax(input_out[0][0].cpu().detach().numpy())
    #     #print(scores)

    #     ranking = np.argsort(scores)
    #     ranking = ranking[::-1]
    #     for i in range(scores.shape[0]):
    #         l = config.id2label[ranking[i]]
    #         s = scores[ranking[i]]
    #         print(f"{i+1}) {l} {np.round(float(s), 4)}")

    #     return topic, config.id2label[ranking[0]]

    # chain of topics
    def generate_cot(text_in):
        tok_text = cot_tokenizer(text_in, return_tensors='pt').to("cuda:0")
        gen_text = cot_model.generate(**tok_text)
        dec_text = cot_tokenizer.decode(gen_text[0], skip_special_tokens=True)
        return dec_text
    
    # topic, sent = extract_topic_sentiment(s_in)
    instruction = "Instruction: Generate a list of topics increasing in specificity to define the subject.\n\n"
    instruction += "Input:[CONTEXT]{}[ENDOFDIALOGUE][QUESTION]The topics defining the input are".format(user_response)
    dec_out = generate_cot(instruction)

    #out_log.write(f"{topic_pref}\t{s_in}\t{dec_out}|{sent}\n")
    out_log.write(f"{topic_pref}\t{s_in}\t{dec_out}\n")
