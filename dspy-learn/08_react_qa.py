import dspy
import os
import inspect

from GLM import GLM


class BasicQA(dspy.Signature):
    """Answer questions with reasoning."""

    question = dspy.InputField()
    answer = dspy.OutputField()


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

    qa = dspy.ReAct(BasicQA, tools=[dspy.Retrieve(k=1)])
    predict_response = qa(question='At My Window was released by which American singer-songwriter?')

    # ReAct会多次调用LLM
    dspy_lm.inspect_history(n=10)
    with open(os.path.join(current_dir, "output/08_qa_ReAct.txt"), 'w') as file:
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



