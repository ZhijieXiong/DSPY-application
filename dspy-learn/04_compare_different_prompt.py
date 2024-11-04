import dspy
import os
import inspect


from util import read_data
from GLM import GLM
from metrics import answer_exact_match


class BasicQA(dspy.Signature):
    """Answer math questions."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="Just final answer which is an integer type")


if __name__ == "__main__":
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    train_data = read_data(os.path.join(current_dir, "gsm8k_data/train.jsonl"))
    test_data = read_data(os.path.join(current_dir, "gsm8k_data/test.jsonl"))

    # api_key = ""
    # 我这里api是存在环境中的，可以指定api_key
    dspy_lm = GLM("zhipu/glm-4-plus")
    dspy.configure(lm=dspy_lm)

    # 使用10个测试样本对比不同prompt方法
    examples = test_data[:10]
    example = examples[0]
    predict = dspy.Predict(BasicQA)
    cot = dspy.ChainOfThought(BasicQA)
    pot = dspy.ProgramOfThought(BasicQA)
    predict_scores = []
    cot_scores = []
    pot_scores = []
    for example in examples:
        predict_response = predict(question=example.question)
        cot_response = cot(question=example.question)
        pot_response = pot(question=example.question)

        predict_score = float(answer_exact_match(example, predict_response))
        cot_score = float(answer_exact_match(example, cot_response))
        pot_score = float(answer_exact_match(example, pot_response))

        predict_scores.append(predict_score)
        cot_scores.append(cot_score)
        pot_scores.append(pot_score)

    with open(os.path.join(current_dir, "output/04_compare_result.txt"), 'w') as file:
        file.write(f"basic predict acc: {sum(predict_scores) / len(predict_scores)}\n")
        file.write(f"cot acc: {sum(cot_scores) / len(cot_scores)}\n")
        file.write(f"pot acc: {sum(pot_scores) / len(pot_scores)}")


