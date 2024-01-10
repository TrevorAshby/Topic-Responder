import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

# set the device
device = torch.device("cuda:0")

model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# quantization_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained(model_id, token='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',torch_dtype=torch.float32)#, device_map='auto'), quantization_config=quantization_config)

model = torch.nn.DataParallel(model, device_ids=[0,1])
model.to(device)
user_in = 'Yeah, their services are good. I\'m just not a fan of intrusive they can be on our personal lives. </s> <s>Google is leading the alphabet subsidiary and will continue to be the Umbrella company for Alphabet internet interest.</s> <s>Did you know Google had hundreds of live goats to cut the grass in the past? \n'
guideline = "The user likes technology. Direct the conversation to one of the following 3 topics: ['technology', 'science', 'technology trends']."
instruction = f"<s>[INST] <<SYS>>\nYou are a chatbot agent. <</SYS>>\nGenerate the next response turn in this conversation: {user_in.replace('</s> <s>', ' ')} Make sure that the response is 1-2 sentences long and compliant with this guideline: {guideline} [/INST]"

prompt = f"{instruction}"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = model.module.generate(**inputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))