import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, T5ForConditionalGeneration

device = torch.device('cuda' if torch.cuda_is_available() else 'cpu')

CONV_HISTORY = [('', '')]
MAX_CONV_TURNS = 4 # make sure that this is an even number

# load the models
topic_tokenizer = AutoTokenizer.from_pretrained('prakharz/DIAL-BART0')
topic_detector = AutoModelForSeq2SeqLM.from_pretrained('trevorashby/topic-detector')

guideliner_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
guideliner = T5ForConditionalGeneration.from_pretrained("trevorashby/guideliner")

blen_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-1B-distill")
blen_model = AutoModelForSeq2SeqLM.from_pretrained("trevorashby/blenderbot-1B-distill")

topic_detector.to(device)
guideliner.to(device)
blen_model.to(device)

topic_detector.to(device)
guideliner.eval()
blen_model.eval()

# alexa skill stuff
invalid_character_mapping = {r"&amp;": "&", "&quot;": '"', "&apos;": "'"}

def get_required_context():
    return required_context

def handle_message(msg, logger=None):
    #your remote module should operate on the text or other context information here
    logger.debug(f'device: {device}')
    response = generate_response(msg['context'], logger=logger)
    
    return response

def generate_response(context, logger=None):
    # index to the last context example, so that most recent user input is grabbed.
    user_response = context[-1]

    # get the topic preferences
    topic_modified_in = ''

    # iterate through the stored conversation history and build an appropriate input string
    for i, turn in enumerate(CONV_HISTORY):
        if turn[1] != '':
            topic_modified_in += 'Human: ' + turn[0] + ' '
        else:
            topic_modified_in += 'Robot: ' + turn[0] + ' '
        if (i < len(CONV_HISTORY)-1) and (i+1 % 2 == 0):
            topic_modified_in += '[ENDOFTURN]'
    topic_in_str = "Instruction: Extract the topic of the last conversation turn, and determine whether the human is interested in it.\n Input: [CONTEXT] " + topic_modified_in + user_response + " [ENDOFDIALOGUE] [QUESTION] Given this conversation provided, the topic and intent is"
    user_input_ids = topic_tokenizer(topic_in_str, max_length=max_length, padding='max_length', return_tensors='pt').input_ids
    topic_pref_example = topic_detector.generate(user_input_ids.to(device), max_new_tokens=128)
    topic_pref = topic_tokenizer.decode(topic_pref_example[0], skip_special_tokens=True)

    # add to the conversation history, if at max, remove 1
    if len(CONV_HISTORY) == MAX_CONV_TURNS:
        CONV_HISTORY.pop(0)
    CONV_HISTORY.append((user_response, topic_pref))

    # generate the guideline with input and topic
    guide_in_str = ''
    topics_combi = ''
    # iterate through the stored conversation history and build an appropriate input string
    for i, turn in enumerate(CONV_HISTORY):
        # B's
        if i+1 % 2 == 0:
            guide_in_str += 'B: ' + turn[0]
        else:
            guide_in_str += 'A: ' + turn[0]
        topics_combi += turn[1]
    
    guide_in_str += '| ' + topics_combi

    in_ids = guideliner_tokenizer(guide_in_str, max_length=max_length, padding='max_length', return_tensors='pt').input_ids
    guideline_example = guideliner.generate(in_ids.to(device), max_new_tokens=50)
    guideline = guideliner_tokenizer.decode(guideline_example[0], skip_special_tokens=True)
    
    # using the guideline generate using one of the models
    generated_response = ''
    blend_in_str = ''
    # iterate through the stored conversation history and build an appropriate input string
    for i, turn in enumerate(CONV_HISTORY):
        blend_in_str += turn[0]
        if (i < len(CONV_HISTORY)-1):
            blend_in_str += '</s> <s>'
    blend_in_str += ' [GUIDELINE] ' + guideline
    blend_in_ids = blen_tokenizer([blend_in_str], max_length=128, return_tensors='pt', truncation=True)
    blend_example = blen_model.generate(**blend_in_ids.to(device), max_length=60)
    blend_response = blen_tokenizer.batch_decode(blend_example, skip_special_tokens=True)[0]
    generated_response = blend_response

    # add to the conversation history, if at max, remove 1
    if len(CONV_HISTORY) == MAX_CONV_TURNS:
        CONV_HISTORY.pop(0)
    CONV_HISTORY.append((generated_response, ''))

    return generated_response