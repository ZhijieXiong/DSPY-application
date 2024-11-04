import chromadb
import inspect
import os
import json

from zhipuai import ZhipuAI


if __name__ == "__main__":
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    with open(os.path.join(current_dir, "airbnb-2023-10k-qca.json"), "r") as f:
        question_context_answers = json.load(f)

    # 数据持久化到chromadb目录下中
    chroma_client = chromadb.PersistentClient(path=os.path.join(current_dir, "chromadb_data"))
    collection = chroma_client.get_or_create_collection(name="contexts_zhipu")
    if collection.count() == 0:
        # 使用api调embedding模型
        client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
        documents = [qca["context"] for qca in question_context_answers]
        ids = [str(i) for i in range(len(question_context_answers))]
        response = client.embeddings.create(
                model="embedding-3",
                input=documents,
                extra_body={
                    "dimensions": 256
                }
        )
        embeddings = [emb.embedding for emb in response.data]
        collection.add(documents=documents, ids=ids, embeddings=embeddings)
