import argparse
import os
import json
import inspect
import dspy


import config
# 不用管显示报错，实际可以运行，手动将remote_llm添加到路径中了的
from remote_llm.GLM import GLM
from remote_llm.BaiLian import BaiLian


from get_mmlu_pro import get_dspy_data
from evaluate import evaluate_from_last
from model import DOTS


class BasicQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="The corresponding option for the answer, like `A` or `F`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="glm-4-plus")
    parser.add_argument("--subset", type=str, default="math",
                        choices=['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                                 'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                                 'psychology', 'history'])
    parser.add_argument("--prompt_method", type=str, default="PoT",
                        choices=("Predict", "CoT", "PoT", "DOTS"))
    parser.add_argument("--num2evaluate", type=int, default=35)
    args = parser.parse_args()

    # 获取当前目录
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    dataset_name = "mmlu_pro"

    # 创建输出目录
    output_dir = os.path.join(current_dir, "output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if args.llm in ["glm-4-plus"]:
        dspy_lm = GLM(f"zhipu/{args.llm}")
    elif args.llm in ["qwen-plus"]:
        dspy_lm = BaiLian(f"bailian/{args.llm}")
    else:
        raise NotImplementedError()
    dspy.configure(lm=dspy_lm)

    # 选择prompt方法
    if args.prompt_method == "Predict":
        predictor = dspy.Predict(BasicQA)
    elif args.prompt_method == "CoT":
        predictor = dspy.ChainOfThought(BasicQA)
    elif args.prompt_method == "PoT":
        predictor = dspy.ProgramOfThought(BasicQA)
    elif args.prompt_method == "DOTS":
        predictor = DOTS()
    else:
        raise NotImplementedError()

    print("loading data ...")
    data_dir = os.path.join(current_dir, "../data/MMLU-PRO")
    val_data, test_data = get_dspy_data(data_dir, args.subset)
    output_path = os.path.join(output_dir, f"output_{dataset_name}_{args.subset}_{args.llm}_{args.prompt_method}.json")
    qa_results = evaluate_from_last(test_data, output_path, predictor, args)

    with open(output_path, 'w') as file:
        file.write(json.dumps(qa_results, indent=4))
