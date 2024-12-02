import argparse
import os
import sys
import json
import inspect
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="qwen-plus")
    parser.add_argument("--prompt", type=str, default="use_concept_prompt")
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cold_start", type=int, default=10,
                        help="如果是0，则每个学生预测全部数据，否则只预测前cold_start个记录")
    args = parser.parse_args()

    # 获取当前目录
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    output_dir = os.path.join(current_dir, "output")

    output_path = os.path.join(output_dir,
                               f"{args.llm}_{args.prompt}_{args.dataset_name}_test_fold_{args.fold}_cold_start_{args.cold_start}.json")
    if not os.path.exists(output_path):
        sys.exit(0)

    with open(output_path, 'r') as file:
        prediction = json.load(file)

    # 创建输出目录
    result_dir = os.path.join(current_dir, "metric")
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    result_path = os.path.join(result_dir, f"{args.llm}_{args.prompt}_{args.dataset_name}_test_fold_{args.fold}_cold_start_{args.cold_start}.json")

    ps = []
    ls = []
    for v in prediction.values():
        pred = []
        for p in v["prediction"]:
            p = p.strip()
            if p in ["0", "1"]:
                pred.append(int(p))
            else:
                # 返回格式不对，随机预测
                pred.append(random.choice([0, 1]))
        ps += pred
        ls += v["ground_truth"]

    evaluation_result = {
        "overall": {
            "acc": accuracy_score(ls, ps),
            "precision": precision_score(ls, ps),
            "recall":  recall_score(ls, ps),
            "f1": f1_score(ls, ps)
        }
    }

    with open(result_path, 'w') as file:
        file.write(json.dumps(evaluation_result, indent=4))
