from torch import cuda, bfloat16
import torch
import transformers
import os,gc,shutil, string
# import gradio as gr
from util.conversation_rag import Conversation_RAG
from util.conversation_cag import Conversation_CAG
from util.index import *
import torch
from model_setup import ModelSetup
import time
import re
from fairseq.models.transformer import TransformerModel
import jieba
from itertools import zip_longest
from copy import deepcopy
import opencc
import soundfile as sf
from TTS.utils.synthesizer import Synthesizer
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    HF_TOKEN = 'INSERT_HF_TOKEN'
    # raise ValueError("HF_TOKEN not found")
    
#--------------
model_path = '../model/llama-3-chinese-8b-instruct-v3'

device = f'cuda:0' if cuda.is_available() else 'cpu'

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    quantization_config=bnb_config,
    token=HF_TOKEN
    # device_map='auto'
).to(device)

model.eval()

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)

#--------------





#-----------替換字詞 start-----------

# 中文 - 定義需要替換的字詞及其對應的替換值
zh_replacements = {"什麼樣子":"什麼樣"}
# 構建正則表達式模式
zh_pattern = re.compile('|'.join(re.escape(key) for key in zh_replacements.keys()))

# 全漢 - 定義需要替換的字詞及其對應的替換值
hl_replacements = {"對不住": "歹勢", "豬哥著": "豬哥亮", "無偌可能":"無可能", "性教育":"", "通定":"通常"}
# 構建正則表達式模式
hl_pattern = re.compile('|'.join(re.escape(key) for key in hl_replacements.keys()))

#-----------替換字詞 end-----------


# OpenCC專用
# s2tw: 簡體中文 -> 繁體中文 (台灣)
# s2twp: 簡體中文 -> 繁體中文 (台灣, 包含慣用詞轉換)
converter = opencc.OpenCC('s2twp') 



#------------------翻譯模型 start------------------

#中文斷詞詞表 路徑待修改
jieba.load_userdict('詞表區/jieba_特定領域詞表.txt') 
jieba.load_userdict('詞表區/icorpus&translation2019zh_chunk0-1&huggingface_詞表.txt')
jieba.load_userdict('詞表區/台語詞表.txt')

#翻譯模型路徑    
model_path = 'model/教育部_教會公報_語料集_punc_zh2hl/'

model_paths = {
    'zh2hl' : 'model/教育部_教會公報_語料集_zh2hl',
    'hl2zh' : 'model/教育部_教會公報_語料集_hl2zh',
    'tl2hl' : 'model/教育部_教會公報_語料集_tl2hl',
    'hl2tl' : 'model/教育部_教會公報_語料集_hl2tl',
}

#check point
ckpt = 'checkpoint_last.pt'

#短文本訓練集-專用模板
zh2hl = TransformerModel.from_pretrained(
  model_path,
  checkpoint_file=ckpt,
  data_name_or_path='training_data/教育部_教會公報_語料集_punc_zh2hl/bin',
  bpe='subword_nmt',
  bpe_codes='BPE/教育部_教會公報_語料集_bpecode.zh',
  device_map="auto"
)

tl2hl = TransformerModel.from_pretrained(
  model_paths['tl2hl'],
  checkpoint_file=ckpt,
  data_name_or_path='教育部_教會公報_語料集_tl2hl/bin',
  bpe='subword_nmt',
  bpe_codes='BPE/教育部_教會公報_語料集_bpecode.tl',
  device_map="auto"
)

hl2zh = TransformerModel.from_pretrained(
  model_paths['hl2zh'],
  checkpoint_file=ckpt,
  data_name_or_path='教育部_教會公報_語料集_hl2zh/bin',
  bpe='subword_nmt',
  bpe_codes='BPE/教育部_教會公報_語料集_2_bpecode.hl',
  device_map="auto"
)

#---------hl2tl模型-------------
model_path_hl2tl = 'model/教育部_教會公報_語料集_hl2tl/'
ckpt_hl2tl = 'checkpoint_last.pt'

hl2tl = TransformerModel.from_pretrained(
    model_path_hl2tl,
    checkpoint_file=ckpt_hl2tl,
    data_name_or_path='training_data/教育部_教會公報_語料集_hl2tl/bin',
    bpe='subword_nmt',
    bpe_codes='BPE/教育部_教會公報_語料集_bpecode.hl',
    device_map=1  # "auto"
)
#------------------------------

model_setup = ModelSetup(HF_TOKEN, "bge-large-zh-v1.5")
success_prompt = model_setup.setup()

#---------zh2hl----------
def translate_and_append(text, model):
    result = []
    non_chinese_pattern = re.compile(r'[^\u4e00-\u9fff]+')
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    while text:
        non_chinese_match = non_chinese_pattern.match(text)
        chinese_match = chinese_pattern.match(text)
        
        if non_chinese_match:
            result.append(text[:non_chinese_match.end()])
            text = text[non_chinese_match.end():]
        
        if chinese_match:
            sentence_chinese = chinese_match.group()
            segmented_words = jieba.cut(sentence_chinese)
            translated_segments = []
            for segment in segmented_words:
                translated_segments.append(model.translate(segment))
            translated_sentence = ''.join(translated_segments)# model.translate(sentence_chinese)
            result.append(translated_sentence)
            text = text[len(sentence_chinese):]
    
    result_text = ''.join(result) #全漢(有標點)
    result_text = hl_pattern.sub(lambda match: hl_replacements[match.group(0)], result_text) #替換翻譯錯的全漢字詞
            
    return result_text


def translate(text, translation_model):
    # 假設這是一個翻譯函數，將文字翻譯成另一種語言
    translated_texts = ""
    translated_text = translate_and_append(text, translation_model)
    
    return translated_text
#------------------------


#---------hl2tl----------
def translate_and_append_hl2tl(text, model):
    result = []
    non_chinese_pattern = re.compile(r'[^\u4e00-\u9fff]+')
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    while text:
        non_chinese_match = non_chinese_pattern.match(text)
        chinese_match = chinese_pattern.match(text)
        
        if non_chinese_match:
            #result.append(text[:non_chinese_match.end()])
            result.append(f" {text[:non_chinese_match.end()]} ") #非中文字，如英文、符號、數字...等，前後都用空格。
            text = text[non_chinese_match.end():]
        
        if chinese_match:
            sentence_chinese = chinese_match.group()
            segmented_words = ' '.join(sentence_chinese) #全漢到台羅，全漢的分詞每個字都用空格去分，不需要用jieba分詞
            translated_segments = [model.translate(segment) for segment in segmented_words]
            translated_sentence = ' '.join(translated_segments) #讓台羅詞每個詞都用空格隔開
            result.append(translated_sentence)
            text = text[len(sentence_chinese):]
    
    result_text = ''.join(result)
    result_text = hl_pattern.sub(lambda match: hl_replacements[match.group(0)], result_text)
            
    return result_text

def translate_hl2tl(text, translation_model):
    translated_text = translate_and_append_hl2tl(text, translation_model)
    return translated_text
#------------------------
def translate_and_append_tl2hl(text,model):
    punctuation_string = list(string.punctuation)
    nonpunc_text = []
    punc_list = []
    st_index = 0
    for index, char in enumerate(text):
        if char == '-':
            continue
        if char in punctuation_string:
            nonpunc_text.append(text[st_index:index])
            st_index = index+1
            punc_list.append(char)
    else:
        if char != '':
            nonpunc_text.append(text[st_index:index])
    if nonpunc_text[-1]=='':
        nonpunc_text.pop(-1)
    
    result = []
    print(nonpunc_text, punc_list)
    for sen, pun in zip_longest(nonpunc_text, punc_list, fillvalue = '.'):
        result.append(model.translate(sen).replace(' ',''))
        result.append(pun)
    
    return ''.join(result)

def translate_and_append_hl2zh(text, model): 
    result = []
    non_chinese_pattern = re.compile(r'[^\u4e00-\u9fff]+')
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    while text:
        non_chinese_match = non_chinese_pattern.match(text)
        chinese_match = chinese_pattern.match(text)
        
        if non_chinese_match:
            result.append(text[:non_chinese_match.end()])
            text = text[non_chinese_match.end():]
        
        if chinese_match:
            sentence_chinese = chinese_match.group()
            # segmented_words = jieba.cut(sentence_chinese)
            # print(segmented_words)
            # translated_segments = []
            # for segment in segmented_words:
            #     translated_segments.append(model.translate(segment))
            # translated_sentence = ''.join(translated_segments)
            result.append(model.translate(sentence_chinese))# translated_sentence
            text = text[len(sentence_chinese):]
    
    result_text = ''.join(result)
    return result_text

#------------------翻譯模型 end------------------

def load_models(hf_token,embedding_model):

    global model_setup

    model_setup = ModelSetup(hf_token, embedding_model)
    success_prompt = model_setup.setup()
    
    # test print
    #print(str(llm))
    #print("成功加載："+str(model_setup.llm))
    
    return success_prompt


def upload_and_create_vector_store(file,embedding_model):
    
    # Save the uploaded file to a permanent location
    file_path = file.name
    split_file_name = file_path.split("/")
    file_name = split_file_name[-1]

    current_folder = os.path.dirname(os.path.abspath(__file__))
    root_folder = os.path.dirname(current_folder)
    data_folder = os.path.join(root_folder, "data")
    permanent_file_path = os.path.join(data_folder, file_name)
    shutil.copy(file.name, permanent_file_path)

    # Access the path of the saved file
    print(f"File saved to: {permanent_file_path}")

    if embedding_model == "bge-large-zh-v1.5":
        embedding_model_repo_id = "../../embedding_model/bge-large-zh-v1.5"
    elif embedding_model == "text2vec-large-chinese":
        embedding_model_repo_id = "../../embedding_model/text2vec-large-chinese"

    index_success_msg = create_vector_store_index(permanent_file_path,embedding_model_repo_id)
    return index_success_msg

augmented_mode = 0
def mode_change(mode_index):
    augmented_mode = mode_index
    mode_tag = ['cag','rag']
    print("mode change : "+mode_tag[augmented_mode])
    return [],[],[]

def get_chat_history(inputs):

    res = []
    for human, ai in inputs:
        res.append(f"問題:{human}\n回答:{ai}")
    return "\n".join(res)

def add_text(history, text):# , history_2
    
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]+')
    for char in text:
        print(char)
        if chinese_pattern.match(char):
            break
    else:
        text = translate_and_append_tl2hl(text, tl2hl)
        text = translate_and_append_hl2zh(text, hl2zh)
        time.sleep(5)
    
    history = history + [[text, None]]
    # history_2 = history_2 + [[text, None]]
    # print(history)
    return history, ""# , history_2

conv_qa = Conversation_RAG()

def bot(history, history_2, history_3,
        instruction="請嘗試使用以下的參考資訊來回答問題，如果在給定的參考資訊中找不到任何與問題相關的資訊，就當作沒看過參考資訊，以你的理解自行回答這個問題，但是回答不能有偽造成分。請用繁體中文回答問題，保持回答生動。",
        temperature=0.2,
        max_new_tokens=2048,
        repetition_penalty=1.1,
        top_k=50,
        top_p=0.9,
        k_context=3,
        num_return_sequences=1,
        ):
    with open('a.txt','a') as f:
        f.write('a')
    
    # streamer = transformers.TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 240429 新增llm_temp = model_setup.llm，用來接取我們用了哪個模型的名稱。
    qa = conv_qa.create_conversation(model,
                             tokenizer,
                             model_setup.vectordb,
                             max_new_tokens=max_new_tokens,
                             temperature=temperature,
                             repetition_penalty=repetition_penalty,
                             top_k=top_k,
                             top_p=top_p,
                             k_context=k_context,
                             num_return_sequences=num_return_sequences,
                             instruction=instruction,
                             llm_temp = "Llama3-8B-Chinese-Chat"

    )

    chat_history_formatted = get_chat_history(history[:-1])
    
    for res in qa.stream(
        {
            'query': history[-1][0],
            'source_documents': chat_history_formatted
        }
    ):
        if "result" in res:
            result = res["result"]
            history_2 = history_2 + [[history[-1][0], None]]
            history_3 = history_3 + [[history[-1][0], None]]
            original_response_簡轉繁 = converter.convert(result)
    
    # res = qa.run(
    #     {
    #         'query': history[-1][0],
    #         'source_documents': chat_history_formatted
    #     }
    # )
    
    # res = qa.invoke(
    #     {
    #         'query': history[-1][0],
    #         'source_documents': chat_history_formatted
    #     }
    # )
    # history[-1][1] = ""
    # history_2[-1] = [[]]
    # history_3[-1] = [[]]
    
    # original_response_原LLM = res['result']
    # original_response_簡轉繁 = converter.convert(original_response_原LLM)
    # history_2 = history_2 + [[history[-1][0], None]]
    # history_3 = history_3 + [[history[-1][0], None]]
    # yield history[-1][1]+res['result'], history_2[:-1], history_3[:-1] 
    
    # history[-1][1] = ""
    # if streamer is not None:
    #     # print('a')
    #     while True:
    #         try:
    #             new_text = next(streamer)
    #             # print(new_text, end="", flush=True)
    #             history[-1][1] += new_text
    #             result += new_text
    #             yield history, history_2[:-1] + [[history_2[-1][0],"全漢轉換中..."]], history_3[:-1] + [[history_3[-1][0],"台羅轉換中..."]]
    #         except StopIteration:
    #             break
    
    #---------terminal 查看用 start---------
    print("-"*50)
    print("*問題：")
    print(str(res['query']))
    print("-"*50)
    print("*回答：")
    print(str(res['result']))
    print('-'*15 +" 檢索到的Source documents " + '-'*15)
    print('*Return Doc數量：' + str(len(res["source_documents"])))
    print(str(res['source_documents']))
    #---------terminal 查看用 end---------
    
    # history_2 = deepcopy(history)  # history的深度copy，用於第二個chatbot框
    # history_3 = deepcopy(history)  # history的深度copy，用於第三個chatbot框

    
    #---------terminal 查看用 start---------
    print(str(history))
    print('******')
    print(str(history_2))
    print('******')
    print(str(history_3))
    #---------terminal 查看用 end---------
    
    
    # original_response_原LLM = res['result'] # 最原始LLM輸出
    
    #簡中 轉 繁中(以防萬一)
    # original_response_簡轉繁 = converter.convert(original_response_原LLM) 
    # #替換翻譯錯的中文字詞
    original_response_簡轉繁 = zh_pattern.sub(lambda match: zh_replacements[match.group(0)], original_response_簡轉繁)
    
    #繁中 轉 全漢
     
    
    #history[-1][1] = original_response_簡轉繁
    #history_2[-1][1]= translated_response
    
    
    
    #translated_history = [[entry[0], translate(entry[1], zh2hl)] if entry[1] else entry for entry in history]
    
     # 立即开始stream显示 original_response_簡轉繁
    history[-1][1] = ""
    for char in original_response_簡轉繁:
        history[-1][1] += char
        time.sleep(0.003)
        yield history, history_2[:-1] + [[history_2[-1][0],"全漢轉換中..."]], history_3[:-1] + [[history_3[-1][0],"台羅轉換中..."]]
    
    #yield history, history_2[:-1] + [[history_2[-1][0],"全漢轉換中..."]]
    
    # 中文轉全漢
    translated_response = translate(original_response_簡轉繁, zh2hl)
    
    history_2[-1][1] = ""
    # 流式输出 translated_response
    for char in translated_response:
        history_2[-1][1] += char
        time.sleep(0.003)
        yield history, history_2, history_3[:-1] + [[history_3[-1][0],"台羅轉換中..."]]
    
    translated_response_hl2tl = translate_hl2tl(translated_response, hl2tl)
    # 後處理翻譯後，還有錯的字
    translated_response_hl2tl = translated_response_hl2tl.replace(" ēnn "," ")
    translated_response_hl2tl = translated_response_hl2tl.replace("𠢕","gâu")
    translated_response_hl2tl = translated_response_hl2tl.replace("𪜶","in")
    translated_response_hl2tl = translated_response_hl2tl.replace("䆀","bái")
    translated_response_hl2tl = translated_response_hl2tl.replace("tsu ah","tsu sìn")
    translated_response_hl2tl = translated_response_hl2tl.replace('-',' ')
    
    history_3[-1][1] = ""
    # 流式输出 translated_response
    for char in translated_response_hl2tl:
        history_3[-1][1] += char
        time.sleep(0.001)
        yield history, history_2, history_3
    
    
    
    translated_history = [[entry[0], translate(entry[1], zh2hl)] if entry[1] else entry for entry in history]
    translated_history_hl2tl = [[entry[0], translate_hl2tl(entry[1], hl2tl)] if entry[1] else entry for entry in history]
    
    print('a')
    return history, translated_history, translated_history_hl2tl


cag_qa = Conversation_CAG(hf_token=HF_TOKEN, kv_path = './kv_cache/long-term_care_knowledges.pt')

def cag_bot(history, history_2, history_3,
        instruction="以你的理解進行對話，但是回答不能有偽造成分。請用繁體中文對話，保持回答簡潔生動。", 
        temperature=0.2,
        max_new_tokens=2048,
        repetition_penalty=1.1,
        top_k=50,
        top_p=0.9,
        num_return_sequences=1):# "你是個幫人解答疑惑的小助手，以你的理解自行回答這個問題，但是回答不能有偽造成分。請用繁體中文回答問題，保持回答簡潔生動，盡量不要使用條列式。",您是一個根據給定上下文給出簡短答案的助手。你是一個根據給定上下文給出簡短答案的助手，請用繁體中文回答問題，保持回答簡潔生動，盡量不要使用條列式。You are an assistant for giving short answers based on given context
    chat_history_formatted = get_chat_history(history[:-1])
    print(chat_history_formatted)
    #question = '你好'
    # cag_qa.load_knowledge_cache(kv_path = )
    cag_qa.clean_up_cache()
    qa = cag_qa.create_conversation(model,
                        tokenizer,
                        llm_temp = "",
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        repetition_penalty=repetition_penalty,
                        top_k=top_k,
                        top_p=top_p,
                        num_return_sequences=num_return_sequences,
                        instruction=instruction,
                        question=history[-1][0],
                        conversation_log = chat_history_formatted)
    # answer = ''
    
    # history_2 = deepcopy(history)
    # history_3 = deepcopy(history)
    
    history_2 = history_2 + [[history[-1][0], None]]
    history_3 = history_3 + [[history[-1][0], None]]
    
    history[-1][1] = ''
    print(history)
    for next_text in qa:
        history[-1][1] = next_text
        yield [history, [[history_2[-1][0],"台漢轉換中..."]], [[history_3[-1][0],"台羅轉換中..."]]]
        # answer = next_text
    # history[-1][1] += next_text
    result = next_text
    
    
    
    # 台漢翻譯
    
    original_response_原LLM = result # 最原始LLM輸出
    
    #簡中 轉 繁中(以防萬一)
    original_response_簡轉繁 = converter.convert(original_response_原LLM) 
    # #替換翻譯錯的中文字詞
    original_response_簡轉繁 = zh_pattern.sub(lambda match: zh_replacements[match.group(0)], original_response_簡轉繁)
    
    translated_response = translate(original_response_簡轉繁, zh2hl)
    
    # print(original_response_原LLM, original_response_簡轉繁, translated_response)
    
    history_2[-1][1] = ""
    # 流式输出 translated_response
    for char in translated_response:
        history_2[-1][1] += char
        time.sleep(0.003)
        yield [history, history_2, history_3[:-1] + [[history_3[-1][0],"台羅轉換中..."]]]
    
    # 台羅翻譯
    translated_response_hl2tl = translate_hl2tl(translated_response, hl2tl).replace('-', ' ')
    
    history_3[-1][1] = ""
    # 流式输出 translated_response
    for char in translated_response_hl2tl:
        history_3[-1][1] += char
        time.sleep(0.001)
        yield [history, history_2, history_3]
    print('b')
    return [history, history_2, history_3]
    
    # history[-1][0]
                             
    
def reset_sys_instruction(instruction):
    default_inst = "請嘗試使用以下的參考資訊來回答問題，如果在給定的參考資訊中找不到任何與問題相關的資訊，就當作沒看過參考資訊，以你的理解自行回答這個問題，但是回答不能有偽造成分。請用繁體中文回答問題，保持回答生動。"
    return default_inst

def bot_answer(augmented_mode, history_1, history_2, history_3):
    if augmented_mode == 1:
        result = bot(history_1, history_2, history_3)
    elif augmented_mode == 0:
        result = cag_bot(history_1, history_2, history_3)
    for a, b, c in result:
        yield a, b, c

def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()
    return None
