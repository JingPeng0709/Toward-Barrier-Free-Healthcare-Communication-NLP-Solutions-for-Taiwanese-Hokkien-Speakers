import os,gc,shutil
from util.conversation_rag import Conversation_RAG
from util.index import *
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class ModelSetup:
    def __init__(self, hf_token, embedding_model):

        self.hf_token = hf_token
        self.embedding_model = embedding_model
        #self.llm = llm

    def setup(self):

        if self.embedding_model == "bge-large-zh-v1.5":
            embedding_model_repo_id = "../embedding_model/bge-large-zh-v1.5"
        elif self.embedding_model == "text2vec-large-chinese":
            embedding_model_repo_id = "../embedding_model/text2vec-large-chinese"


        # if self.llm == "chinese-alpaca-2-13b-16k":
        #     llm_repo_id = "../../model/chinese-alpaca-2-13b-16k"
        # elif self.llm == "chinese-alpaca-2-13b":
        #     llm_repo_id = "../../model/chinese-alpaca-2-13b"
        # elif self.llm == "TAIDE-LX-7B-Chat":
        #     llm_repo_id = "../../model/TAIDE-LX-7B-Chat"
        # elif self.llm == "Llama3-8B-Chinese-Chat":
        #     llm_repo_id = "../../model/Llama3-8B-Chinese-Chat"
        # elif self.llm == "Llama3-TAIDE-LX-8B-Chat-Alpha1":
        #     llm_repo_id = "../../model/Llama3-TAIDE-LX-8B-Chat-Alpha1"

        conv_rag = Conversation_RAG(self.hf_token,
                                    embedding_model_repo_id)

        self.vectordb = conv_rag.load_model_and_tokenizer()
        return "Setup Complete"