#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   vectordb.py
@Time    :   2023/12/24 15:42:29
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   本文件封装了向量数据库的对外接口
'''

from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm
import os
import erniebot
from dotenv import load_dotenv, find_dotenv
from ernie_embedding import ErnieEmbeddings

# 可以调用的向量数据库对象
class VectorDB:

    def __init__(self, db_path:str, embedding:str, embedding_path:str=None) -> None:
        # args:
        # db_path: 本地向量数据库路径
        # embedding: 使用的 embedding 模型
        # embedding_path: embedding 模型本地路径，使用 API 则保持 None
        self.db_path = db_path
        self.embedding = embedding
        if self.embedding == "ernie":
            # 目前只支持百度的 Embedding
            # 加载文心 Key
            _ = load_dotenv(find_dotenv())
            env_file = os.environ
            erniebot.api_type = 'aistudio'
            erniebot.access_token = env_file["access_token"]

            # 加载文心词向量模型
            embeddings = ErnieEmbeddings(access_token=env_file["access_token"])
        else:
            raise KeyError("No such embedding model.")
        
        # 加载向量数据库
        # 目前只支持 4 祝福
        db_key_lst = ["4"]
        self.dbs = {}
        for one_db_key in db_key_lst:
            vectorstore = Chroma(persist_directory=os.path.join(self.db_path, one_db_key), embedding_function=embeddings)
            self.dbs[one_db_key] = vectorstore

    # 调用向量数据库进行检索
    def search(self, search_input:dict[str,any], top_k:int=5)->list:
        # args:
        # search_input: 字典，包括键 'query' 和 'type',query 为查询语句，type 为查询类型（即细分场景，使用 int 序号）
        # top_k: 返回的结果数
        # return: 列表格式的检索结果，按照相关度从高到低排序
        # 先确定要读取的数据库
        db_key = str(search_input["type"])
        if db_key not in self.dbs:
            raise KeyError("No such database.")
        
        # 进行检索
        vectorstore = self.dbs[db_key]
        contentList = vectorstore.similarity_search_with_score(search_input["query"], top_k=top_k)

        return contentList

    
