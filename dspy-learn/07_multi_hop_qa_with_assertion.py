import dspy
import os
import inspect

from dsp.utils import deduplicate
# from dspy.datasets.gsm8k import gsm8k_metric
from GLM import GLM


class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")


class GenerateSearchQuery(dspy.Signature):
    """Write a simple search query that will help answer a complex question."""

    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    query = dspy.OutputField()


def validate_query_distinction_local(previous_queries, query):
    """check if query is distinct from previous queries"""
    if not previous_queries:
        return True
    if dspy.evaluate.answer_exact_match_str(query, previous_queries, frac=0.8):
        return False
    return True


def all_queries_distinct(prev_queries):
    query_distinct = True
    for i, query in enumerate(prev_queries):
        if not validate_query_distinction_local(prev_queries[:i], query):
            query_distinct = False
            break
    return query_distinct


# def all_queries_distinct(queries):
#     if len(queries) <= 1:
#         return True
#     else:
#         # 循环检验所有query是否不同
#         for i in range(len(queries) - 2):
#             current_query = queries[i]
#             rest_queries = queries[i:]
#             if not validate_query_distinction_local(rest_queries, current_query):
#                 return False
#         return True


class SimplifiedBaleenAssertions(dspy.Module):
    def __init__(self, passages_per_hop=2, max_hops=2):
        super().__init__()
        self.generate_query = [dspy.ChainOfThought(GenerateSearchQuery) for _ in range(max_hops)]
        self.retrieve = dspy.Retrieve(k=passages_per_hop)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
        self.max_hops = max_hops

    def forward(self, question):
        context = []
        prev_queries = [question]

        for hop in range(self.max_hops):
            query = self.generate_query[hop](context=context, question=question).query
            # Suggest不会中断，dspy.Assert会中断
            dspy.Suggest(
                len(query) <= 100,
                "Query should be short and less than 100 characters",
                target_module=self.generate_query
            )
            dspy.Suggest(
                validate_query_distinction_local(prev_queries, query),
                "Query should be distinct from: "
                + "; ".join(f"{i+1}) {q}" for i, q in enumerate(prev_queries)),
                target_module=self.generate_query
            )

            prev_queries.append(query)
            passages = self.retrieve(query).passages
            context = deduplicate(context + passages)

        # 会报错，可能新的版本中没有这个passed_suggestions属性
        # if all_queries_distinct(prev_queries):
        #     # 记录通过Suggest的数量
        #     self.passed_suggestions += 1

        pred = self.generate_answer(context=context, question=question)
        pred = dspy.Prediction(context=context, answer=pred.answer)
        return pred


# 7-official-examples/23-qa/hotpotqa_with_assertions.ipynb
if __name__ == "__main__":
    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)

    # api_key = ""
    # 我这里api是存在环境中的，可以指定api_key
    dspy_lm = GLM("zhipu/glm-4-plus")
    dspy.configure(lm=dspy_lm)

    # Define a retrieval model server to send retrieval requests to
    colbertv2_wiki17_abstracts = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')

    # Configure retrieval server internally
    dspy.settings.configure(rm=colbertv2_wiki17_abstracts)

    qa = SimplifiedBaleenAssertions()
    predict_response = qa(question='At My Window was released by which American singer-songwriter?')

    # 因为是多跳，所以会多次调用LLM
    dspy_lm.inspect_history(n=10)
    with open(os.path.join(current_dir, "output/07_qa_assertion.txt"), 'w') as file:
        for item in dspy_lm.history:
            messages = item["messages"] or [{"role": "user", "content": item["prompt"]}]
            outputs = item["outputs"]
            timestamp = item.get("timestamp", "Unknown time")

            file.write(f"[{timestamp}]" + "\n\n")

            for msg in messages:
                file.write(f"**{msg['role'].capitalize()} message:**" + "\n")
                file.write(msg["content"].strip() + "\n")
                file.write("\n" + "\n")

            file.write("**Response:**" + "\n")
            file.write(outputs[0].strip() + "\n")
            file.write("\n\n\n" + "\n")



