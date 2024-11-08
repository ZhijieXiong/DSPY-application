import json
import dspy
import os
import inspect
import argparse


from get_big_bench_hard_data import get_dspy_data
from remote_llm.GLM import GLM
from evaluate import evaluate_from_last
from model import DOTS


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="glm-4-plus")
    parser.add_argument("--subset", type=str, default="causal_judgement",
                        choices=["tracking_shuffled_objects_seven_objects", "salient_translation_error_detection",
                                 "tracking_shuffled_objects_three_objects", "geometric_shapes", "object_counting",
                                 "word_sorting", "logical_deduction_five_objects", "hyperbaton", "sports_understanding",
                                 "logical_deduction_seven_objects", "multistep_arithmetic_two", "ruin_names",
                                 "causal_judgement", "logical_deduction_three_objects", "formal_fallacies", "snarks",
                                 "boolean_expressions", "reasoning_about_colored_objects", "dyck_languages", "navigate",
                                 "disambiguation_qa", "temporal_sequences", "web_of_lies",
                                 "tracking_shuffled_objects_five_objects",
                                 "penguins_in_a_table", "movie_recommendation", "date_understanding"]
                        )
    parser.add_argument("--prompt_method", type=str, default="DOTS",
                        choices=("Predict", "CoT", "PoT", "DOTS"))
    parser.add_argument("--num2evaluate", type=int, default=3)
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
    if args.llm in ["glm-4-plus"]:
        dspy_lm = GLM("zhipu/glm-4-plus")
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
    elif args.prompt_method == "DOTS":
        predictor = DOTS()
    else:
        raise NotImplementedError()

    print("loading data ...")
    data_dir = os.path.join(current_dir, "../data/BIG-BENCH-HARD")
    data = get_dspy_data(data_dir, args.subset)
    output_path = os.path.join(output_dir, f"output_{dataset_name}_{args.subset}_{args.llm}_{args.prompt_method}.json")
    qa_results = evaluate_from_last(data, output_path, predictor, args)

    with open(output_path, 'w') as file:
        file.write(json.dumps(qa_results, indent=4))
