import json
import os
import random

import dspy
from tqdm import tqdm


class InitState(dspy.Signature):
    """根据学生练习第一道习题的情况，初始化学生的知识状态，并使用布鲁姆教学体系认知领域中的6个层级表示学生在不同知识点上的掌握程度
布鲁姆教学体系认知领域中的6个层级如下：
记忆（Remembering）：识记和回忆知识，能够准确复述或列举事实和概念的基本知识。例如，记住定义、公式、事件等。
理解（Understanding）：理解知识的含义，能够解释、比较或归纳信息。例如，将一个概念用自己的话表达出来，解释主要思想。
应用（Applying）：能够将学到的知识应用到新的情境中，解决实际问题。例如，在一个真实场景中应用某个数学公式。
分析（Analyzing）：将知识分解为各组成部分，理解其结构和关系，识别知识的关键因素和层次。例如，分析论点的逻辑结构或分解复杂问题。
评价（Evaluating）：对知识进行评价和判断，能够根据标准或标准对信息或方法进行判断，形成评价意见。例如，判断某一方法的有效性或依据证据评估论点的优劣。
创造（Creating）：整合知识和技能，创造新的产品或提出新的想法，能够在新的或不确定的情境中整合信息。例如，设计实验方案、撰写论文、发明新方法。"""
    question = dspy.InputField(desc="学生练习的第一道习题")
    correctness = dspy.InputField(desc="学生作答情况，0表示`做错`，1表示`做错`")
    state = dspy.OutputField(desc="学生初始的知识状态，格式为`知识点名称`：`认知层级`, `更具体的描述`，每个知识点占一行")


class UpdateState(dspy.Signature):
    """根据学生上一时刻的知识状态，以及学生此时作答习题的情况，更新学生的知识状态，并使用布鲁姆教学体系认知领域中的6个层级表示学生在不同知识点上的掌握程度
布鲁姆教学体系认知领域中的6个层级如下：
记忆（Remembering）：识记和回忆知识，能够准确复述或列举事实和概念的基本知识。例如，记住定义、公式、事件等。
理解（Understanding）：理解知识的含义，能够解释、比较或归纳信息。例如，将一个概念用自己的话表达出来，解释主要思想。
应用（Applying）：能够将学到的知识应用到新的情境中，解决实际问题。例如，在一个真实场景中应用某个数学公式。
分析（Analyzing）：将知识分解为各组成部分，理解其结构和关系，识别知识的关键因素和层次。例如，分析论点的逻辑结构或分解复杂问题。
评价（Evaluating）：对知识进行评价和判断，能够根据标准或标准对信息或方法进行判断，形成评价意见。例如，判断某一方法的有效性或依据证据评估论点的优劣。
创造（Creating）：整合知识和技能，创造新的产品或提出新的想法，能够在新的或不确定的情境中整合信息。例如，设计实验方案、撰写论文、发明新方法。"""
    state_last = dspy.InputField(desc="学生上一时刻的知识状态")
    question = dspy.InputField(desc="学生此时练习的习题")
    correctness = dspy.InputField(desc="学生作答情况，0表示`做错`，1表示`做错`")
    updated_state = dspy.OutputField(desc="学生更新后的知识状态，格式为`知识点名称`：`认知层级`, `更具体的描述`，每个知识点占一行")


class PredictAnswerCorrectness(dspy.Signature):
    """根据学生上一时刻的知识状态（使用布鲁姆教学体系认知领域中的6个层级表示），预测学生能否做对指定习题，并给出解释
布鲁姆教学体系认知领域中的6个层级如下：
记忆（Remembering）：识记和回忆知识，能够准确复述或列举事实和概念的基本知识。例如，记住定义、公式、事件等。
理解（Understanding）：理解知识的含义，能够解释、比较或归纳信息。例如，将一个概念用自己的话表达出来，解释主要思想。
应用（Applying）：能够将学到的知识应用到新的情境中，解决实际问题。例如，在一个真实场景中应用某个数学公式。
分析（Analyzing）：将知识分解为各组成部分，理解其结构和关系，识别知识的关键因素和层次。例如，分析论点的逻辑结构或分解复杂问题。
评价（Evaluating）：对知识进行评价和判断，能够根据标准或标准对信息或方法进行判断，形成评价意见。例如，判断某一方法的有效性或依据证据评估论点的优劣。
创造（Creating）：整合知识和技能，创造新的产品或提出新的想法，能够在新的或不确定的情境中整合信息。例如，设计实验方案、撰写论文、发明新方法。"""
    state_last = dspy.InputField(desc="学生上一时刻的知识状态，格式为`知识点名称`：`认知层级`, `更具体的描述`")
    question = dspy.InputField(desc="此时练习的习题，待预测")
    correctness = dspy.OutputField(desc="预测的学生作答情况，0表示`做错`，1表示`做错`，严格要求只回答0或1")
    prediction_explanation = dspy.OutputField(desc="对预测的解释")


def base_prompt_evaluate(kt_data, question_data, output_path, args):
    if not os.path.exists(output_path):
        prediction = {}
    else:
        with open(output_path, 'r') as file:
            prediction = json.load(file)

    num_evaluated = 0
    progress_bar = tqdm(total=args.num2evaluate)
    for item_data in kt_data:
        if num_evaluated >= args.num2evaluate:
            break

        user_id = str(item_data["user_id"])

        if args.cold_start <= 0:
            seq_len = item_data["seq_len"]
        else:
            seq_len = min(item_data["seq_len"], args.cold_start+1)

        if user_id in prediction:
            num_predicted = len(prediction[user_id]["prediction"])
            if num_predicted >= (seq_len - 1):
                continue
            correctness = item_data["correct_seq"][num_predicted+1:seq_len]
            question2predict = item_data["question_seq"][num_predicted+1:seq_len]
        else:
            correctness = item_data["correct_seq"][1:seq_len]
            question2predict = item_data["question_seq"][1:seq_len]
            prediction[user_id] = {
                "prediction": [],
                "ground_truth": [],
            }
            try:
                prediction[user_id]["state_last"] = dspy.Predict(InitState)(
                    question=question_data[str(item_data["question_seq"][0])]["content"],
                    correctness=item_data["correct_seq"][0]
                ).state
            except:
                # 预测出错，则设置初始知识状态为空
                prediction[user_id]["state_last"] = "No practice records, unknown student knowledge status"

        for q_id, c in zip(question2predict, correctness):
            question = question_data[str(q_id)]["content"]
            state_last = prediction[user_id]["state_last"]
            try:
                pred = dspy.Predict(PredictAnswerCorrectness)(state_last=state_last, question=question).correctness
            except:
                # 预测出错，随机判断
                pred = str(random.choice([0, 1]))
            prediction[user_id]["prediction"].append(pred)
            prediction[user_id]["ground_truth"].append(c)
            try:
                prediction[user_id]["state_last"] = dspy.Predict(UpdateState)(
                    state_last=state_last, question=question, correctness=c
                ).updated_state
            except:
                # 预测出错，则不更新状态
                pass
            num_evaluated += 1
            progress_bar.update(1)

            if num_evaluated >= args.num2evaluate:
                break
    progress_bar.close()

    return prediction
