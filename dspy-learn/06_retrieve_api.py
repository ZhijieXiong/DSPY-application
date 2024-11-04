import os
import inspect
from dspy.retrieve.chromadb_rm import ChromadbRM
from zhipuai import ZhipuAI


def embedding_function(texts):
    client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
    response = client.embeddings.create(
        model="embedding-3",
        input=texts,
        extra_body={
            "dimensions": 256
        }
    )
    embeddings = [emb.embedding for emb in response.data]
    return embeddings


if __name__ == "__main__":
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)

    retriever_model = ChromadbRM(
        'contexts_zhipu',
        os.path.join(current_dir, "chromadb_data"),
        embedding_function=embedding_function,
        k=5
    )

    results = retriever_model("By what percentage did Airbnb's net income increase in 2023 compared to the prior year?", k=5)

    for result in results:
        print("Document:", result.long_text, "\n")

