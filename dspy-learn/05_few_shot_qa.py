import dspy
import os
import inspect

from dspy.teleprompt import LabeledFewShot

from util import read_data


class BasicQA(dspy.Signature):
    """Answer questions with reasoning."""

    question = dspy.InputField()
    answer = dspy.OutputField()


if __name__ == "__main__":
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    train_data = read_data(os.path.join(current_dir, "gsm8k_data/train.jsonl"))
    test_data = read_data(os.path.join(current_dir, "gsm8k_data/test.jsonl"))

    api_key = os.getenv("GLM_API_KEY")
    dspy_lm = dspy.LM('deepseek/glm4', api_key=api_key)
    dspy.configure(lm=dspy_lm)

    examples = test_data[:10]
    example = examples[0]

    # Define teleprompter
    teleprompter = LabeledFewShot(k=2)
    # LabeledFewShot会随机选择k个样本作为例子
    predict = teleprompter.compile(student=dspy.ChainOfThought(BasicQA), trainset=train_data)
    predict_response = predict(question=example.question)

    # 打印prompt和返回结果，并存入文件
    dspy_lm.inspect_history(1)
    item = dspy_lm.history[-1]
    with open(os.path.join(current_dir, "output/05_qa_few_shot.txt"), 'w') as file:
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

