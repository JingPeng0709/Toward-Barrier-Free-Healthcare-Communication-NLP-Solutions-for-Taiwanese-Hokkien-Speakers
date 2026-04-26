import torch
import os
# import cag.dataset as cagds
# import cag.similarity as cagsim
from time import time
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM, TextStreamer, TextIteratorStreamer
from transformers.cache_utils import DynamicCache
import logging 
# from config import ConfigName, set_config
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from dotenv import load_dotenv
load_dotenv()

torch.serialization.add_safe_globals([DynamicCache])
torch.serialization.add_safe_globals([set])

os.environ["CUDA_VISIBLE_DEVICES"]='0,1'
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = 'INSERT_HF_TOKEN'
    # raise ValueError("HF_TOKEN not found")
    
# callbacks = [StreamingStdOutCallbackHandler()]

class Conversation_CAG:
    def __init__(self, hf_token=HF_TOKEN, 
                 # embedding_model_repo_id="../../../embedding_model/bge-large-zh-v1.5",
                kv_path = 'path/to/kv_cache'):
        
        self.hf_token = hf_token
        self.kv_cache = torch.load(kv_path, map_location='cuda:0')
        self.kv_len = self.kv_cache.key_cache[0].shape[-2]
        # self.embedding_model_repo_id = embedding_model_repo_id
        
    def load_knowledge_cache(self, kv_path = 'path/to/kv_cache'):
        kv_cache = torch.load(kv_path, map_location='cuda:0')
        self.kv_cache = kv_cache
        #self.kv_len = kv.key_cache[0].shape[-2]
        
    def clean_up_cache(self):
        for i in range(len(self.kv_cache.key_cache)):
            self.kv_cache.key_cache[i] = self.kv_cache.key_cache[i][:, :, :self.kv_len, :]
            self.kv_cache.value_cache[i] = self.kv_cache.value_cache[i][:, :, :self.kv_len, :]
        
    def create_conversation(self, model, tokenizer, llm_temp, max_new_tokens=512, temperature=0.2, repetition_penalty=1.1, top_k=50, top_p=0.9, num_return_sequences=1, instruction="以你的理解並自行回答這個問題，但是回答不能有偽造成分。請用繁體中文回答問題，保持回答簡潔生動。", question="你好嗎?", conversation_log = ''):
        
        # streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        embed_device = model.model.embed_tokens.weight.device
#         system_instruction = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>
# <|start_header_id|>user<|end_header_id|>\n
# {question}
# <|eot_id|><|start_header_id|>assistant<|end_header_id|>\n
# """
        system_instruction = f"""{question}

回答:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"""
#         if conversation_log == '':
#             system_instruction = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}
#             <|start_header_id|>user<|end_header_id|>
#             問題:
#             {question}
            
#             回答:
#             <|eot_id|>
#             <|start_header_id|>assistant<|end_header_id|>
#             """
#         else:
#             #context='以下是對話紀錄'
#             # question = "請介紹YunTech one電動車"
#             # print('-'*20+'\n'+question+'\n'+'-'*20)
#             instruction = "請基於上述對話內容，以你的理解並自行回答我的問題，但是回答不能有偽造成分。請用繁體中文回答問題，保持回答簡潔生動。"
#             system_instruction = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
#             你是一個對話小助手
            
#             以下是我們的對話內容:
#             {conversation_log}
            
#             {instruction}
#             <|start_header_id|>user<|end_header_id|>
            
#             問題:
#             {question}
            
#             回答:
#             <|eot_id|>
#             <|start_header_id|>assistant<|end_header_id|>
#             """
#         print(system_instruction)
        input_ids = tokenizer.encode(system_instruction, return_tensors="pt").to(embed_device)
        origin_ids = input_ids
        input_ids = input_ids.to(embed_device)
        output_ids = input_ids.clone()
        next_token = input_ids
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = model(
                    input_ids=next_token, 
                    past_key_values=self.kv_cache,
                    use_cache=True
                )
                next_token_logits = outputs.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1).unsqueeze(-1)
                next_token = next_token.to(embed_device)

                self.kv_cache = outputs.past_key_values

                output_ids = torch.cat([output_ids, next_token], dim=1)

                yield tokenizer.decode(output_ids[:, origin_ids.shape[-1]:][0], 
                                          skip_special_tokens=True, 
                                          temperature=0.2)

                if next_token.item() in [model.config.eos_token_id]:
                    # yield 'end0'
                    break
                
        # generate_text = 
