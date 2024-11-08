import argparse
import os
import sys
import json
import inspect
import dspy
from tqdm import tqdm
from dspy.evaluate import answer_exact_match


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="glm-4-plus")
    parser.add_argument("--dataset_name", type=str, default="big_bench_hard")
    parser.add_argument("--subset", type=str, default="causal_judgement")
    parser.add_argument("--prompt_method", type=str, default="PoT",
                        choices=("Predict", "CoT", "PoT", "DOTS"))
    args = parser.parse_args()

    # 获取当前目录
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)

    # 创建输出目录
    output_dir = os.path.join(current_dir, "output")
    if not os.path.exists(output_dir):
        sys.exit(0)

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

    result_path = os.path.join(result_dir, f"metric_string_match_{args.dataset_name}_{args.llm}.json")
    if not os.path.exists(result_path):
        evaluation_result = {args.subset: {}}
    else:
        with open(result_path, 'r') as file:
            evaluation_result = json.load(file)

    scores = []
    print("start calculate metrics")
    for qa in tqdm(qa_results.values()):
        if args.dataset_name == "mmlu_pro":
            score = float(qa["answer"].lower() == qa["response"].strip().strip("(").strip(")").strip().lower())
        elif args.dataset_name == "big_bench_hard":
            score = answer_exact_match(
                dspy.Example(answer=qa["answer"]),
                dspy.Example(answer=qa["response"]),
                frac=0.8
            )
        else:
            raise NotImplementedError()
        scores.append(score)

    if args.subset not in evaluation_result:
        evaluation_result[args.subset] = {}
    evaluation_result[args.subset][args.prompt_method] = {
        "acc": sum(scores) / len(scores)
    }

    with open(result_path, 'w') as file:
        file.write(json.dumps(evaluation_result, indent=4))
