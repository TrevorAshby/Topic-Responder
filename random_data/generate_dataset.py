# THIS FILE IS USED TO GENERATE DATA USED TO FINE-TUNE GUIDELINER AND RESPONDER. MAKE SURE TO CLEAN DATA BEFORE TRAINING ON IT

import openai
from multiprocessing import Pool
import time

# create a file names 'api_key.txt' and on the first line place your openai API key.
openai.api_key = open('./api_key.txt', "r").read()
COMMAND_FILE = '' # INSERT THE PATH TO THE CHATGPT PROMPT YOU WANT TO USE

save_dir = './'
print("===== You loaded the {} API key =====".format(openai.api_key))

def sample_conv(file_id):
    example='''{}'''.format(open(COMMAND_FILE, 'r').read())
    print("===== THE PROMPT YOU ARE USING =====")
    print(example)
    print("====================================")

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[
                #{"role": "system", "content": "Imagine you are an intelligent screenwriter. You would like to write a short conversation. It could include one or several topics. Besides the conversation, you also need to annotate the conversation with topics and if the human is interested in each topic. High-level topic refers to broad categories encompassing a wide range of subtopics and experiences; middle-level topics are more specific subtopics that fall within the broader categories of high-level topics, tending to be more focused and concrete; low-level refers to a specific person, place, or thing. Additionally, you want to write a guideline of how to best respond based upon the human interests."},
                #{"role": "assistant", "content": "Alright, I'm imagining that I'm an intelligent screenwriter!"},
                {"role": "user", "content": example},
                #{"role": "assistant", "content": "Understand. I will generate new examples."},
                #{"role": "user", "content": "Generate a new conversation with random topics. The topic part needs to follow the JSON format constantly."},
        ]
    )
    #print(response)
    t = int(time.time())
    with open(save_dir + '/' + str(file_id) + '_' + str(t) + '.txt', 'w') as f:
        f.write(response['choices'][0]['message']['content'])
    time.sleep(1)

ls = [*range(110)]
with Pool(110) as p:
    p.map(sample_conv, ls)
