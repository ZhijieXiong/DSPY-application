import dspy
from remote_llm.GLM import GLM


class AnalysisSign(dspy.Signature):
    """Choose the most suitable problem-solving strategy for a given question. The actions that need to be decided are as follows,
analysis_action: Analyze the question, with options including (A) "Rewrite the question to make it clearer", (B) "Decompose the questions into multiple subtasks to solve the sub-question", and (C) "Do nothing";
solution_action: Solve the question, with options including (A) "Chain of Thought, i.e., step-by-step reasoning with language", (B) "Program of Thought, i.e., using a program to solve the question", and (C) "Directly solving the question";
do_self_verify: Choose whether to self verify the answer, with options of "Yes" or "No"."""
    question = dspy.InputField()
    analysis_action = dspy.OutputField(desc="`A` or `B` or `C`")
    solution_action = dspy.OutputField(desc="`A` or `B` or `C`")
    do_self_verify = dspy.OutputField(desc="`Yes` or `No`")
    explanation = dspy.OutputField(desc="Explanation of choosing such a problem-solving strategy")


class RewriteSign(dspy.Signature):
    """In this step, you need to reveal the core_question of original_question with only a simple sentence, and extract useful_information from the original_question that is useful for answering the core_question."""
    original_question = dspy.InputField()
    core_question = dspy.OutputField()
    useful_information = dspy.OutputField()


class DecomposeSign(dspy.Signature):
    """In this step, you need to break down the original_question to decomposed_questions in your own words. Analyze how you can decompose the question into smaller, more manageable sub-tasks. Pay attention to small details, nuances, notes and examples in the question description."""
    original_question = dspy.InputField()
    decomposed_questions = dspy.OutputField(desc="decomposed sub-tasks")


class RewriteQASign(dspy.Signature):
    """Answer the question based on the rewrote_question"""
    question = dspy.InputField()
    rewrote_question = dspy.InputField()
    answer = dspy.OutputField()


class DecomposeQASign(dspy.Signature):
    """Answer the question based on the decomposed_question"""
    question = dspy.InputField()
    decomposed_question = dspy.InputField()
    answer = dspy.OutputField()


class SelfVerifyRewriteSign(dspy.Signature):
    """In this step, you need to carefully verify the correctness of the generated_answer with natural language."""
    question = dspy.InputField()
    rewrote_question = dspy.InputField()
    generated_answer = dspy.InputField()
    verification_result = dspy.OutputField(desc="`Pass` or `Fail`")
    feedback = dspy.OutputField(desc="Error reason (if the result of verification is that the generated_answer is wrong)")


class SelfVerifyDecomposeSign(dspy.Signature):
    """In this step, you need to carefully verify the correctness of the generated_answer with natural language."""
    question = dspy.InputField()
    decomposed_question = dspy.InputField()
    generated_answer = dspy.InputField()
    verification_result = dspy.OutputField(desc="`Pass` or `Fail`")
    feedback = dspy.OutputField(desc="Error reason (if the result of verification is that the generated_answer is wrong)")


class SelfVerifyDirectSign(dspy.Signature):
    """In this step, you need to carefully verify the correctness of the generated_answer with natural language."""
    question = dspy.InputField()
    generated_answer = dspy.InputField()
    verification_result = dspy.OutputField(desc="`Pass` or `Fail`")
    feedback = dspy.OutputField(desc="Error reason (if the result of verification is that the generated_answer is wrong)")


class RetryDirectQASign(dspy.Signature):
    """Re answer the question based on the previous_answer and  verification_feedback from the self verification results."""
    question = dspy.InputField()
    previous_answer = dspy.InputField(desc="Previous answer results")
    verification_feedback = dspy.InputField(desc="The feedback from the self verification results")
    answer = dspy.OutputField(desc="The result of re answering")


class RetryRewriteQASign(dspy.Signature):
    """Re answer the question based on the previous_answer and  verification_feedback from the self verification results."""
    question = dspy.InputField()
    rewrote_question = dspy.InputField()
    previous_answer = dspy.InputField(desc="Previous answer results")
    verification_feedback = dspy.InputField(desc="The feedback from the self verification results")
    answer = dspy.OutputField(desc="The result of re answering")


class RetryDecomposeQASign(dspy.Signature):
    """Re answer the question based on the previous_answer and  verification_feedback from the self verification results."""
    question = dspy.InputField()
    decomposed_question = dspy.InputField()
    previous_answer = dspy.InputField(desc="Previous answer results")
    verification_feedback = dspy.InputField(desc="The feedback from the self verification results")
    answer = dspy.OutputField(desc="The result of re answering")


class DOTS(dspy.Module):
    def __init__(self, max_verify=2):
        super().__init__()
        self.planner = dspy.Predict(AnalysisSign)
        self.rewrite = dspy.Predict(RewriteSign)
        self.decompose = dspy.Predict(DecomposeSign)
        self.max_verify = max_verify

        self.solution_module = {
            "A": dspy.ChainOfThought,
            "B": dspy.ProgramOfThought,
            "C": dspy.Predict
        }

    def forward(self, question):
        plan_response = self.planner(question=question)
        analysis_action = plan_response.analysis_action.strip().strip("(").strip(")").strip().upper()
        solution_action = plan_response.solution_action.strip().strip("(").strip(")").strip().upper()
        do_self_verify = plan_response.do_self_verify.strip().lower().strip()

        solution_keys = {
            "question": question
        }
        self_verify_keys = {
            "question": question
        }
        retry_qa_keys = {
            "question": question
        }
        if analysis_action == "A":
            analysis_response = self.rewrite(original_question=question)
            core_question = analysis_response.core_question
            useful_information = analysis_action.useful_information
            rewrote_question = f"core_question: {core_question}\nuseful_information: \n{useful_information}"
            solution_sign = RewriteQASign
            self_verify_sign = SelfVerifyRewriteSign
            retry_qa_sign = RetryRewriteQASign
            solution_keys["rewrote_question"] = rewrote_question
            self_verify_keys["rewrote_question"] = rewrote_question
            retry_qa_keys["rewrote_question"] = rewrote_question
        elif analysis_action == "B":
            analysis_response = self.decompose(original_question=question)
            solution_sign = DecomposeQASign
            self_verify_sign = SelfVerifyDecomposeSign
            retry_qa_sign = RetryDecomposeQASign
            solution_keys["decomposed_question"] = analysis_response.decomposed_questions
            self_verify_keys["decomposed_question"] = analysis_response.decomposed_questions
            retry_qa_keys["decomposed_question"] = analysis_response.decomposed_questions
        else:
            solution_sign = "question -> answer"
            self_verify_sign = SelfVerifyDirectSign
            retry_qa_sign = RetryDirectQASign

        solution_response = self.solution_module[solution_action](solution_sign)(**solution_keys)
        answer = solution_response.answer

        if do_self_verify == "yes":
            verification_result = "fail"
            count_verify = 0
            while verification_result != "pass" and count_verify < self.max_verify:
                self_verify_keys["generated_answer"] = answer
                verification_response = dspy.Predict(self_verify_sign)(**self_verify_keys)
                count_verify += 1
                verification_result = verification_response.verification_result.strip().lower().strip()
                if verification_result == "fail":
                    retry_qa_keys["previous_answer"] = answer
                    retry_qa_keys["verification_feedback"] = verification_response.feedback
                    solution_response = self.solution_module[solution_action](retry_qa_sign)(**retry_qa_keys)
                    answer = solution_response.answer

        return solution_response


if __name__ == "__main__":
    dspy_lm = GLM("zhipu/glm-4-plus")
    dspy.configure(lm=dspy_lm)
    DOTS()(question="""How would a typical person answer each of the following questions about causation?
Long ago, when John was only 17 years old, he got a job working for a large manufacturing company. He started out working on an assembly line for minimum wage, but after a few years at the company, he was given a choice between two line manager positions. He could stay in the woodwork division, which is where he was currently working. Or he could move to the plastics division. John was unsure what to do because he liked working in the woodwork division, but he also thought it might be worth trying something different. He finally decided to switch to the plastics division and try something new. For the last 30 years, John has worked as a production line supervisor in the plastics division. After the first year there, the plastics division was moved to a different building with more space. Unfortunately, through the many years he worked there, John was exposed to asbestos, a highly carcinogenic substance. Most of the plastics division was quite safe, but the small part in which John worked was exposed to asbestos fibers. And now, although John has never smoked a cigarette in his life and otherwise lives a healthy lifestyle, he has a highly progressed and incurable case of lung cancer at the age of 50. John had seen three cancer specialists, all of whom confirmed the worst: that, except for pain, John's cancer was untreatable and he was absolutely certain to die from it very soon (the doctors estimated no more than 2 months). Yesterday, while John was in the hospital for a routine medical appointment, a new nurse accidentally administered the wrong medication to him. John was allergic to the drug and he immediately went into shock and experienced cardiac arrest (a heart attack). Doctors attempted to resuscitate him but he died minutes after the medication was administered. Did misadministration of medication cause John's premature death?
Options:
- Yes
- No""")
    dspy_lm.inspect_history(n=10)

