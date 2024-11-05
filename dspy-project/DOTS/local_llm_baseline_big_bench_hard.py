import json
import dspy
import os
import inspect
import argparse


from get_big_bench_hard_data import get_dspy_data
from local_llm.LocalLLM import LocalLLM
from evaluate import evaluate_from_last


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="glm-4-plus")
    parser.add_argument("--subset", type=str, default="causal_judgement")
    parser.add_argument("--prompt_method", type=str, default="CoT",
                        choices=("Predict", "CoT", "PoT"))
    parser.add_argument("--num2evaluate", type=int, default=5)
    args = parser.parse_args()

    # 获取当前目录
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    dataset_name = "big_bench_hard"

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

    # 选择prompt方法
    if args.prompt_method == "Predict":
        predictor = dspy.Predict("question -> answer")
    elif args.prompt_method == "CoT":
        predictor = dspy.ChainOfThought("question -> answer")
    elif args.prompt_method == "PoT":
        predictor = dspy.ProgramOfThought("question -> answer")
    else:
        raise NotImplementedError()

    print("loading data ...")
    data_dir = os.path.join(current_dir, "../data/BIG-BENCH-HARD")
    data = get_dspy_data(data_dir, args.subset)
    output_path = os.path.join(output_dir, f"output_{dataset_name}_{args.subset}_{args.llm}_{args.prompt_method}.json")
    qa_results = evaluate_from_last(data, output_path, predictor, args)

    with open(output_path, 'w') as file:
        file.write(json.dumps(qa_results, indent=4))

    result_path = os.path.join(output_dir, f"result_{dataset_name}_{args.llm}.json")
    if not os.path.exists(result_path):
        evaluation_result = {args.subset: {}}
    else:
        with open(result_path, 'r') as file:
            evaluation_result = json.load(file)

    scores = []
    for qa in qa_results.values():
        score = float(qa["answer"].lower() == qa["response"].strip().strip("(").strip(")").strip().lower())
        scores.append(score)

    if args.subset not in evaluation_result:
        evaluation_result[args.subset] = {}
    evaluation_result[args.subset][args.prompt_method] = {
        "acc": sum(scores) / len(scores)
    }

    with open(result_path, 'w') as file:
        file.write(json.dumps(evaluation_result, indent=4))
