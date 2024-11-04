import os
import inspect
from dspy.retrieve.chromadb_rm import ChromadbRM

import chromadb.utils.embedding_functions as ef


if __name__ == "__main__":
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)

    embedding_function = ef.DefaultEmbeddingFunction()
    retriever_model = ChromadbRM(
        'contexts',
        os.path.join(current_dir, "chromadb_data"),
        embedding_function=embedding_function,
        k=5
    )

    results = retriever_model("What was Airbnb's revenue in 2023?", k=5)

    for result in results:
        print("Document:", result.long_text, "\n")

