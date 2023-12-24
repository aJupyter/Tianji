#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   ernie_qa.py
@Time    :   2023/12/17 15:51:16
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于文心一言和本地知识库的问答系统
'''

import erniebot
import os
from dotenv import load_dotenv, find_dotenv
from ernie_embedding import ErnieEmbeddings
from langchain.vectorstores import Chroma


def readDB(access_token, persist_directory="../../data/vectordb/chroma"):
    assert os.path.isdir(persist_directory)
    # # Embed and store splits
    embeddings=ErnieEmbeddings(access_token=access_token)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    return vectorstore

## 基于用户的query去搜索库中相近的文档
def searchSimDocs(query,vectorstore,top_k=5,scoreThershold=5):
    packs=vectorstore.similarity_search_with_score(query,k=top_k)
    contentList=[]

    for pack in packs:
        doc,score=pack
        if score<scoreThershold:##好像设置5，基本都会返回
            contentList.append(doc.page_content)

    # print('content',contentList)
    return contentList

def packPrompt(query,contentList):
    prompt="你是善于总结归纳并结合文本回答问题的文本助理。请使用以下检索到的上下文来回答问题。如果你不知道答案，就说你不知道。最多使用三句话，并保持答案简洁。问题为：\n"+query+" \n上下文：\n"+'\n'.join(contentList) +" \n 答案:"
    return prompt

def singleQuery(prompt,model='ernie-bot'):
    response = erniebot.ChatCompletion.create(
        model=model,
        messages=[{
            'role': 'user',
            'content': prompt
        }])
    print('response',response)
    try:
        resFlag=response['rcode']
    except:        
        resFlag=response['code']
    if resFlag==200:
        try:
            data=response['body']
        except:
            data=response

        result=response['result']
            
        usedToken=data['usage']['total_tokens']
    else:
        result=""
        usedToken=-1
    return result,usedToken

class ErnieQA:

    def __init__(self, db_file_path:str="../../data/vectordb/chroma"):
        # 加载文心 Key
        _ = load_dotenv(find_dotenv())
        env_file = os.environ
        erniebot.api_type = 'aistudio'
        erniebot.access_token = env_file["access_token"]
        self.vectordb = readDB(env_file["access_token"], db_file_path)

    def answer(self, query:str, top_k:int=5, model:str='ernie-bot-4'):
        contentList=searchSimDocs(query,self.vectordb,top_k=top_k)
        # print('contentList',contentList)
        prompt=packPrompt(query,contentList)
        res,usedToken=singleQuery(prompt,model=model)
        return res, contentList, usedToken
