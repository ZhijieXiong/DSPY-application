import dspy


def answer_exact_match(example, pred, trace=None):
    try:
        # 答案都是int类型，但是有些答案以类似40,000这种形式给出
        return int(example.answer.strip().replace(",", "")) == int(pred.answer.strip().replace(",", ""))
    except:
        return False


class ReasoningAssess(dspy.Signature):
    """Assess the quality of a reasoning along the specified dimension."""
    
    # gt: ground truth
    question = dspy.InputField()
    gt_reasoning = dspy.InputField()
    pred_reasoning = dspy.InputField()
    # 评估答案的维度
    assessment_question = dspy.InputField()
    # 评估结果
    assessment_answer = dspy.OutputField(desc="Yes or No")


class AnswerAssess(dspy.Signature):
    """Determine whether the model's answer is numerically consistent with the standard answer? (The accuracy error is within 0.001)"""

    # gt: ground truth
    gt_answer = dspy.InputField(desc="The standard answer")
    pred_answer = dspy.InputField(desc="The model's answer")
    # 评估结果
    assessment_answer = dspy.OutputField(desc="Yes or No")


def reasoning_match(example, pred, trace=None, llm_assess=None):
    question, gt_answer, gt_reasoning = example.question, example.answer, example.reasoning
    pred_answer, pred_reasoning = pred.answer, pred.reasoning

    simplicity_assess = (f"The reasoning of {question} is {pred_reasoning}. "
                         f"Is the reasoning process concise and easy to understand?")
    consistency_assess = (f"The reasoning process of the standard answer is {gt_reasoning}. "
                          f"The reasoning process of the model answer is {pred_reasoning}. "
                          f"Is the reasoning process of the model's answer consistent with that of the standard answer?")

    if llm_assess is not None:
        # 使用指定模型评估
        with dspy.context(lm=llm_assess):
            correctness = dspy.Predict(AnswerAssess)(gt_answer=gt_answer, pred_answer=pred_answer)
            simplicity = dspy.Predict(ReasoningAssess)(question=question, gt_reasoning=gt_reasoning,
                                                       pred_reasoning=pred_reasoning, assessment_question=simplicity_assess)
            consistency = dspy.Predict(ReasoningAssess)(question=question, gt_reasoning=gt_reasoning,
                                                        pred_reasoning=pred_reasoning, assessment_question=consistency_assess)
    else:
        correctness = dspy.Predict(AnswerAssess)(gt_answer=gt_answer, pred_answer=pred_answer)
        simplicity = dspy.Predict(ReasoningAssess)(question=question, gt_reasoning=gt_reasoning,
                                                   pred_reasoning=pred_reasoning, assessment_question=simplicity_assess)
        consistency = dspy.Predict(ReasoningAssess)(question=question, gt_reasoning=gt_reasoning,
                                                    pred_reasoning=pred_reasoning, assessment_question=consistency_assess)

    correctness, simplicity, consistency = \
        [m.assessment_answer.lower() == 'yes' for m in [correctness, simplicity, consistency]]

    answer_score = float(correctness)
    reasoning_score = (simplicity + consistency) / 2

    return answer_score, reasoning_score
