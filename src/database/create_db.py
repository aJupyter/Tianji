#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   create_db.py
@Time    :   2023/12/17 14:47:08
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   基于 txt 源文件构造向量数据库
'''

# 首先导入所需第三方库
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from tqdm import tqdm
import os
import erniebot
from dotenv import load_dotenv, find_dotenv
from ernie_embedding import ErnieEmbeddings

# 加载文件函数
def get_text(dir_path):
    # args：dir_path，目标文件夹路径
    # 首先调用上文定义的函数得到目标文件路径列表
    file_lst = os.listdir(dir_path)
    # docs 存放加载之后的纯文本对象
    docs = []
    # 遍历所有目标文件
    for one_file in tqdm(file_lst):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(os.path.join(dir_path, one_file))
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(os.path.join(dir_path, one_file))
        else:
            # 如果是不符合条件的文件，直接跳过
            continue
        docs.extend(loader.load())
    return docs


if __name__ == "__main__":
    
    # 目标文件夹
    tar_dir = [
        "../../data/corpus/4祝福"
    ]

    # 加载目标文件
    docs = []
    for dir_path in tar_dir:
        docs.extend(get_text(dir_path))

    # 对文本进行分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, separators='\n\n')
    split_docs = text_splitter.split_documents(docs)

    # 加载文心 Key
    _ = load_dotenv(find_dotenv())
    env_file = os.environ
    erniebot.api_type = 'aistudio'
    erniebot.access_token = env_file["access_token"]

    # 加载文心词向量模型
    embeddings = ErnieEmbeddings(access_token=env_file["access_token"])

    # 构建向量数据库
    # 定义持久化路径
    persist_directory = '../../data/vectordb/chroma/4'
    # 加载数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
    )
    # 将加载的向量数据库持久化到磁盘上
    vectordb.persist()