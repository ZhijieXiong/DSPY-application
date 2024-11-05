import argparse
import os
import json
import inspect
import dspy


from get_mmlu_pro import get_dspy_data
from local_llm.LocalLLM import LocalLLM
from evaluate import evaluate_from_last


class BasicQA(dspy.Signature):
    question = dspy.InputField()
    answer = dspy.OutputField(desc="The corresponding option for the answer, like `A` or `F`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="llama3-8b-instruct")
    parser.add_argument("--subset", type=str, default="math",
                        choices=['computer science', 'math', 'chemistry', 'engineering', 'law', 'biology',
                                 'health', 'physics', 'business', 'philosophy', 'economics', 'other',
                                 'psychology', 'history'])
    parser.add_argument("--prompt_method", type=str, default="CoT",
                        choices=("Predict", "CoT", "PoT"))
    parser.add_argument("--num2evaluate", type=int, default=50)
    args = parser.parse_args()

    # 获取当前目录
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    dataset_name = "mmlu_pro"

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
        predictor = dspy.Predict(BasicQA)
    elif args.prompt_method == "CoT":
        predictor = dspy.ChainOfThought(BasicQA)
    elif args.prompt_method == "PoT":
        predictor = dspy.ProgramOfThought(BasicQA)
    else:
        raise NotImplementedError()

    print("loading data ...")
    data_dir = os.path.join(current_dir, "../data/MMLU-PRO")
    val_data, test_data = get_dspy_data(data_dir, args.subset)
    output_path = os.path.join(output_dir, f"output_{dataset_name}_{args.subset}_{args.llm}_{args.prompt_method}.json")
    qa_results = evaluate_from_last(test_data, output_path, predictor, args)

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
