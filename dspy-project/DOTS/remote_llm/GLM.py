import os
import dspy
from datetime import datetime
from zhipuai import ZhipuAI


def _green(text: str, end: str = "\n"):
    return "\x1b[32m" + str(text).lstrip() + "\x1b[0m" + end


def _red(text: str, end: str = "\n"):
    return "\x1b[31m" + str(text) + "\x1b[0m" + end


def _inspect_history(history, n: int = 1):
    """Prints the last n prompts and their completions."""

    for item in history[-n:]:
        messages = item["messages"]
        outputs = item["outputs"]
        timestamp = item.get("timestamp", "Unknown time")

        print("\n\n\n")
        print("\x1b[34m" + f"[{timestamp}]" + "\x1b[0m" + "\n")

        for msg in messages:
            print(_red(f"{msg['role'].capitalize()} message:"))
            print(msg["content"].strip())
            print("\n")

        print(_red("Response:"))
        print(_green(outputs[0].strip()))

        if len(outputs) > 1:
            choices_text = f" \t (and {len(outputs)-1} other completions)"
            print(_red(choices_text, end=""))

    print("\n\n\n")


class GLM(dspy.LM):
    def __init__(self, model, api_key=None, endpoint=None, **kwargs):
        self.endpoint = endpoint
        self.history = []

        super().__init__(model, **kwargs)

        if api_key is None:
            self.model = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
        else:
            self.model = ZhipuAI(api_key=api_key)

    def __call__(self, prompt=None, messages=None, **kwargs):
        # Custom chat model working for text completion model
        prompt = '\n\n'.join([x['content'] for x in messages] + ['BEGIN RESPONSE:'])

        completions = self.model.chat.completions.create(
            model="glm-4-plus",  # 填写需要调用的模型编码
            messages=messages,
        )

        outputs = [completions.choices[0].message.content]
        self.history.append({"messages": messages, "outputs": outputs, "timestamp": datetime.now().isoformat()})

        # Must return a list of strings
        return outputs

    def inspect_history(self, n: int = 1):
        _inspect_history(self.history, n)


if __name__ == "__main__":
    lm = GLM("zhipu/glm-4-plus")
    dspy.configure(lm=lm)

    qa = dspy.ChainOfThought("question->answer")
    answer = qa(question="What is the capital of France?")
    lm.inspect_history(n=1)
