import argparse
import inspect
import dspy
from tqdm import tqdm


from utils import *
from base_prompt_evaluate import base_prompt_evaluate

import config
# 不用管显示报错，实际可以运行，手动将remote_llm添加到路径中了的
from remote_llm.GLM import GLM
from remote_llm.BaiLian import BaiLian


class TranslateSign(dspy.Signature):
    """将question_zh翻译成英文，要求语句通顺，无语法错误"""
    question_zh = dspy.InputField()
    question_en = dspy.OutputField()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", type=str, default="qwen-plus")
    parser.add_argument("--dataset_name", type=str, default="moocradar-C_746997")
    parser.add_argument("--num2evaluate", type=int, default=550)
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

    data_dir = os.path.join(current_dir, f"data/{args.dataset_name}")
    question_translated_path = os.path.join(data_dir, f"question_translated.json")
    if not os.path.exists(question_translated_path):
        question_translated = {}
    else:
        with open(question_translated_path, 'r') as file:
            question_translated = json.load(file)

    question_data = load_json(os.path.join(data_dir, f"question.json"))
    progress_bar = tqdm(total=args.num2evaluate)
    translate = dspy.Predict(TranslateSign)
    num_evaluated = 0
    for q_id, q_value in question_data.items():
        if num_evaluated >= args.num2evaluate:
            break

        if q_id in question_translated.keys():
            continue

        q_content = q_value["content"]
        q_type = q_value["type"]
        question_zh = f"{q_type}\n{q_content}"
        if q_type.strip() in ["单选题", "多选题"]:
            q_options = "\n".join([f"{option_key}: {option_value}" for option_key, option_value in q_value["option"].items()])
            question_zh = f"{question_zh}\n选项：\n{q_options}"
        try:
            response = translate(question_zh=question_zh)
            question_translated[q_id] = {
                "content": response.question_en,
                "question": response.question_en,
                "type": q_type,
                "answer": q_value["answer"],
                "options": q_value.get("option", {}),
                "concepts": q_value["concepts"]
            }
            num_evaluated += 1
            progress_bar.update(1)
        except:
            break

    progress_bar.close()
    write_json(question_translated, question_translated_path)




