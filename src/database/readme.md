# 数据库使用说明

**最新更新日期 2023.12.24**

注意：① 目前仅支持百度文心 Embedding；② 目前支持的向量数据库仅为4（祝福语）

## 使用步骤

1. 在 .env 中填入百度 API Key

2. 从 vectordb.py 中导入 VectorDB 对象并实例化

3. 已构建的初版向量数据库路径在 `../../data/vectordb/chroma/`

4. 调用 VectorDB 对象的 search 方法即可检索，细节查阅 test.ipynb