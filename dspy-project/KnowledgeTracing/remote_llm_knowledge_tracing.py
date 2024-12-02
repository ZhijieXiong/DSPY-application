import argparse
import inspect
import dspy


from utils import *
from base_prompt_evaluate import base_prompt_evaluate
from use_concept_prompt_evaluate import use_concept_prompt_evaluate

import config
# 不用管显示报错，实际可以运行，手动将remote_llm添加到路径中了的
from remote_llm.GLM import GLM
from remote_llm.BaiLian import BaiLian


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="qwen-plus")
    parser.add_argument("--dataset_name", type=str, default="xes3g5m")
    parser.add_argument("--prompt", type=str, default="base_prompt",
                        choices=("base_prompt", "use_concept_prompt"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--cold_start", type=int, default=10,
                        help="如果是0，则每个学生预测全部数据，否则只预测前cold_start个记录")
    parser.add_argument("--num2evaluate", type=int, default=50)
    args = parser.parse_args()

    # 获取当前目录
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)

    # 选择LLM
    if args.llm in ["glm-4-plus"]:
        dspy_lm = GLM(f"zhipu/{args.llm}")
    elif args.llm in ["qwen-plus"]:
        dspy_lm = BaiLian(f"bailian/{args.llm}")
    else:
        raise NotImplementedError()
    dspy.configure(lm=dspy_lm)

    # 转换一下user id
    if not os.path.exists(os.path.join(current_dir, f"data/{args.dataset_name}/test_uid_map_fold_{args.fold}.txt")):
        kt_data = read_preprocessed_file(os.path.join(current_dir, f"data/test_fold_{args.fold}.txt"))
        for i, item_data in enumerate(kt_data):
            item_data["user_id"] = i
        write2file(kt_data, os.path.join(current_dir, f"data/{args.dataset_name}/test_uid_map_fold_{args.fold}.txt"))
    else:
        kt_data = read_preprocessed_file(os.path.join(current_dir, f"data/{args.dataset_name}/test_uid_map_fold_{args.fold}.txt"))

    question_data = load_json(os.path.join(current_dir, f"data/{args.dataset_name}/question.json"))

    # 创建输出目录
    output_dir = os.path.join(current_dir, "output")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(current_dir, f"output/{args.llm}_{args.prompt}_{args.dataset_name}_test_fold_{args.fold}_cold_start_{args.cold_start}.json")

    if args.prompt == "base_prompt":
        prediction = base_prompt_evaluate(
            kt_data, question_data, output_path, args
        )
    elif args.prompt == "use_concept_prompt":
        prediction = use_concept_prompt_evaluate(
            kt_data, question_data, output_path, args
        )
    else:
        raise NotImplementedError()

    with open(output_path, 'w', encoding="utf-8") as file:
        file.write(json.dumps(prediction, indent=2, ensure_ascii=False))
