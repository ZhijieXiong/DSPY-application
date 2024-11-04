import dspy
import os
import inspect
from util import read_data
from metrics import reasoning_match

from GLM import GLM


class BasicQA(dspy.Signature):
    """Answer questions with reasoning."""

    question = dspy.InputField()
    answer = dspy.OutputField()


if __name__ == "__main__":
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    train_data = read_data(os.path.join(current_dir, "gsm8k_data/train.jsonl"))
    test_data = read_data(os.path.join(current_dir, "gsm8k_data/test.jsonl"))

    # api_key = ""
    # 我这里api是存在环境中的，可以指定api_key
    dspy_lm = GLM("zhipu/glm-4-plus")
    dspy.configure(lm=dspy_lm)

    predict = dspy.ChainOfThought(BasicQA)
    answer_scores = []
    reasoning_scores = []
    examples = test_data[:2]
    for example in examples:
        # 最基本的Predictor输出
        predict_response = predict(question=example.question)
        answer_score, reasoning_score = reasoning_match(example, predict_response)
        answer_scores.append(answer_score)
        reasoning_scores.append(reasoning_score)
    print(f"answer acc: f{sum(answer_scores) / len(answer_scores)}")
    print(f"reasoning acc: f{sum(reasoning_scores) / len(reasoning_scores)}")


