import inspect
import os
import json
import dspy
import random
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


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class RAG(dspy.Module):
    def __init__(self, data_path, collection_name, num_passages=3):
        super().__init__()
        self.retrieve = ChromadbRM(collection_name, data_path,
                                   embedding_function=embedding_function, k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question):
        context = [r["long_text"] for r in self.retrieve(question)]
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)


# 参考实现：https://docs.parea.ai/tutorials/dspy-rag-trace-evaluate/tutorial
if __name__ == "__main__":
    deepseek_key = "sk-7933a1ebfa9d41d3acc855ae7cd9aa10"
    dspy_lm = dspy.LM('deepseek/deepseek-chat', api_key=deepseek_key)
    dspy.configure(lm=dspy_lm)

    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    with open(os.path.join(current_dir, "airbnb-2023-10k-qca.json"), "r") as f:
        question_context_answers = json.load(f)

    qca_dataset = []
    for qca in question_context_answers:
        # Tell DSPy that the 'question' field is the input. Any other fields are labels and/or metadata.
        qca_dataset.append(
            dspy.Example(question=qca["question"], answer=qca["answer"], golden_context=qca["context"]).with_inputs(
                "question"))

    random.seed(2024)
    random.shuffle(qca_dataset)
    train_set = qca_dataset[: int(0.7 * len(qca_dataset))]
    test_set = qca_dataset[int(0.7 * len(qca_dataset)):]

    # 使用一个例子看一下rag的prompt
    rag = RAG(collection_name="contexts_zhipu", data_path=os.path.join(current_dir, "chromadb_data"))
    train_ex = train_set[0]
    predict_response = rag(question=train_ex.question)
    item = dspy_lm.history[-1]
    with open(os.path.join(current_dir, "output/06_qa_RAG.txt"), 'w') as file:
        messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        file.write(f"[{timestamp}]" + "\n\n")

        for msg in messages:
            file.write(f"**{msg['role'].capitalize()} message:**" + "\n")
            file.write(msg["content"].strip() + "\n")
            file.write("\n" + "\n")

        file.write("**Response:**" + "\n")
        file.write(outputs[0].strip() + "\n")
        file.write("\n\n\n" + "\n")



