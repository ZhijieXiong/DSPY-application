import argparse
import os
import sys
import json
import inspect
import dspy
from tqdm import tqdm

import config
# 不用管显示报错，实际可以运行，手动将local_llm添加到路径中了的
from local_llm.LocalLLM import LocalLLM


class AnswerAssess(dspy.Signature):
    """Determine whether the model's answer is consistent with the standard answer?
Choose the correctness criteria based on the question requirements: (1) Complete string match (2) Complete semantic match (3) Semantic consistency
Note: Only the final judgment result is needed, no process is required"""
    question = dspy.InputField()
    # gt: ground truth
    gt_answer = dspy.InputField(desc="The standard answer")
    pred_answer = dspy.InputField(desc="The model's answer")
    # 评估结果
    assessment = dspy.OutputField(desc="Yes or No")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="llama3-8b-instruct")
    parser.add_argument("--dataset_name", type=str, default="mmlu_pro")
    parser.add_argument("--subset", type=str, default="math")
    parser.add_argument("--prompt_method", type=str, default="CoT",
                        choices=("Predict", "CoT", "PoT"))
    args = parser.parse_args()

    # 获取当前目录
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)

    # 创建输出目录
    output_dir = os.path.join(current_dir, "output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # 选择LLM
    if args.llm in ["llama2-7b-chat", "llama2-7b-chat-hf", "llama3-8b-instruct"]:
        dspy_lm = LocalLLM(args.llm)
    else:
        raise NotImplementedError()
    dspy.configure(lm=dspy_lm)

    output_path = os.path.join(output_dir,
                               f"output_{args.dataset_name}_{args.subset}_{args.llm}_{args.prompt_method}.json")
    if not os.path.exists(output_path):
        sys.exit(0)

    with open(output_path, 'r') as file:
        qa_results = json.load(file)

    if len(qa_results) == 0:
        sys.exit(0)

    # 创建输出目录
    result_dir = os.path.join(current_dir, "metric")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    result_path = os.path.join(result_dir, f"metric_{args.llm}_assess_{args.dataset_name}_{args.llm}.json")
    if not os.path.exists(result_path):
        evaluation_result = {args.subset: {}}
    else:
        with open(result_path, 'r') as file:
            evaluation_result = json.load(file)

    scores = []
    print("start calculate metrics")
    for qa in tqdm(qa_results.values()):
        try:
            e = dspy.Predict(AnswerAssess)(question=qa["question"], gt_answer=qa["answer"], pred_answer=qa["response"])
            score = float(e.assessment.strip().lower() == 'yes')
        except:
            # 默认做错
            score = 0.0
        scores.append(score)

    if args.subset not in evaluation_result:
        evaluation_result[args.subset] = {}
    evaluation_result[args.subset][args.prompt_method] = {
        "acc": sum(scores) / len(scores)
    }

    with open(result_path, 'w') as file:
        file.write(json.dumps(evaluation_result, indent=4))
