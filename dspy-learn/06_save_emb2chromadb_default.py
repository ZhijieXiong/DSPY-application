import chromadb
import inspect
import os
import json


if __name__ == "__main__":
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    with open(os.path.join(current_dir, "airbnb-2023-10k-qca.json"), "r") as f:
        question_context_answers = json.load(f)

    # 数据持久化到chromadb目录下中
    chroma_client = chromadb.PersistentClient(path=os.path.join(current_dir, "chromadb_data"))
    collection = chroma_client.get_or_create_collection(name="contexts")
    if collection.count() == 0:
        # 默认使用all-MiniLM-L6-v2生成embedding
        collection.add(documents=[qca["context"] for qca in question_context_answers],
                       ids=[str(i) for i in range(len(question_context_answers))])
