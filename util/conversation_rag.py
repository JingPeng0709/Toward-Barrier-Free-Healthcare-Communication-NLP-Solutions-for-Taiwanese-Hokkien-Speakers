from torch import cuda, bfloat16
import transformers
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
# from langchain.llms import HuggingFacePipeline
from langchain_community.llms import HuggingFacePipeline
from huggingface_hub import login
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from transformers import pipeline, TextStreamer
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer, pipeline
from threading import Thread
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

callbacks = [StreamingStdOutCallbackHandler()]
# streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

class Conversation_RAG:
    def __init__(self, hf_token = "", embedding_model_repo_id="../../../embedding_model/bge-large-zh-v1.5"):
        
        self.hf_token = hf_token
        self.embedding_model_repo_id = embedding_model_repo_id
        #self.llm_repo_id = llm_repo_id

    def load_model_and_tokenizer(self):

        embedding_model = HuggingFaceEmbeddings(model_name=self.embedding_model_repo_id)
        vectordb = FAISS.load_local("./db/faiss_index", embedding_model, allow_dangerous_deserialization = True)

        #login(token=self.hf_token)

#         device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

#         bnb_config = transformers.BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_quant_type='nf4',
#             bnb_4bit_use_double_quant=True,
#             bnb_4bit_compute_dtype=bfloat16
#         )

#         model = transformers.AutoModelForCausalLM.from_pretrained(
#             self.llm_repo_id,
#             trust_remote_code=True,
#             quantization_config=bnb_config,
#             device_map='auto'
#         )
        
#         model.eval()

#         tokenizer = transformers.AutoTokenizer.from_pretrained(self.llm_repo_id, use_fast=False, trust_remote_code=True)
        return vectordb

    def create_conversation(self, model, tokenizer, vectordb, llm_temp, max_new_tokens=512, temperature=0.2, repetition_penalty=1.1, top_k=50, top_p=0.9, k_context=3, num_return_sequences=1, instruction="請嘗試使用以下的參考資訊來回答問題，如果在給定的參考資訊中找不到任何與問題相關的資訊，就當作沒看過參考資訊，以你的理解自行回答這個問題，但是回答不能有偽造成分。請用繁體中文回答問題，保持回答生動。"):
        
        #streamer = TextStreamer(tokenizer, skip_prompt=True)
        #streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        # streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generate_text = transformers.pipeline(
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,  # langchain expects the full text
            task='text-generation',
            temperature=temperature,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
            max_new_tokens=max_new_tokens,  # mex number of tokens to generate in the output default:512
            repetition_penalty=repetition_penalty,  # without this output begins repeating
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            
        )

        llm = HuggingFacePipeline(pipeline=generate_text,
            callbacks=callbacks,)
        
        
        
        
        #llm_temp是自己加的，為了接取並判斷我們使用了哪個模型
        if llm_temp == "Llama3-8B-Chinese-Chat" or llm_temp == "Llama3-TAIDE-LX-8B-Chat-Alpha1":
            system_instruction = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}"
            template = system_instruction + """
            <|eot_id|><|start_header_id|>user<|end_header_id|>
            
            問題:
            {question}

            參考資訊:
            ```
            {context}
            ```

            問題:
            {question}
            
            回答:
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
            print("666666 你成功使用Llama-3模板了!!")
        else:
            system_instruction = f"<s>[INST] <<SYS>>\n{instruction}\n<</SYS>>\n\n"
            template = system_instruction + """
            問題:
            {question}

            上下文片段:
            ```
            {context}
            ```

            問題:
            {question}
            [/INST]
            """
            print("7777777")

        

        #QCA_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
        
        
        # 對話 chain
        #memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True, output_key='answer')        
        #memory=memory,
        
        '''
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type='stuff',
            retriever=vectordb.as_retriever(search_kwargs={"k": k_context}),
            combine_docs_chain_kwargs={"prompt": QCA_PROMPT},
            get_chat_history=lambda h: h,
            verbose=True
            return_source_documents=True,
        )
        '''
        
        PROMPT = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": PROMPT}
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectordb.as_retriever(search_kwargs={"k": k_context}),
            chain_type_kwargs=chain_type_kwargs,
            verbose=True
        )
          
        
        # for new_text in streamer:
        #     history[-1][1]  += new_text
        #     time.sleep(0.05)
        #     yield history
        
        
        
        
        
        
        #print(str(qa))
        
        
        
        return qa

