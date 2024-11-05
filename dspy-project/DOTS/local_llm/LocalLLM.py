import dspy
import torch
import os
import transformers
from datetime import datetime
from transformers import AutoTokenizer

from .model.llama2.generation import Llama as Llama2
from .model.llama3.generation import Llama as Llama3


MODEL_NAME2TYPE = {
    "llama2-7b-chat": Llama2,
    "llama3-8b-instruct": Llama3
}


LLM_DIR = "/data/xiongzj/LLM"
LLMs = ["llama2-7b-chat", "llama2-7b-chat-hf", "llama3-8b-instruct"]
LLM_DICT = {
    llm_name: {
        "path": os.path.join(LLM_DIR, llm_name), 
        "is_hf": "hf" in llm_name
    } 
    for llm_name in LLMs
}



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


class LocalLLM(dspy.LM):
    def __init__(self, llm_name, endpoint=None, **kwargs):
        self.endpoint = endpoint
        self.history = []
        super().__init__(llm_name, **kwargs)

        self.llm_name = llm_name
        if LLM_DICT[llm_name]["is_hf"]:
            self.model = transformers.pipeline(
                "text-generation",
                model=LLM_DICT[llm_name]["path"],
                torch_dtype=torch.float16,
                # Do not use `device_map` AND `device` at the same time as they will conflict
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(LLM_DICT[llm_name]["path"])
        else:
            self.model = MODEL_NAME2TYPE[llm_name].build(
                ckpt_dir=LLM_DICT[llm_name]["path"],
                max_batch_size=4,
                tokenizer_path=os.path.join(LLM_DICT[llm_name]["path"], "tokenizer.model"),
                max_seq_len=8192*2,
            )
            self.tokenizer = None


    def __call__(self, prompt=None, messages=None, **kwargs):
        # Custom chat model working for text completion model
        prompt = '\n\n'.join([x['content'] for x in messages] + ['BEGIN RESPONSE:'])
        if LLM_DICT[self.llm_name]["is_hf"]:
            sequences = self.model(
                prompt,
                # do_sample=True,
                # top_k=5,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id,
                truncation=False,
                max_length=8192,
            )
            generated_text = sequences[0]["generated_text"]
        else:
            generated_results = self.model.chat_completion(
                [messages],
                max_gen_len=None,
                # temperature=params["temperature"],
                # top_p=params["top_p"],
            )
            generated_text = generated_results[0]['generation']['content']
        
        outputs = [generated_text]
        self.history.append({"messages": messages, "outputs": outputs, "timestamp": datetime.now().isoformat()})

        # Must return a list of strings
        return outputs

    def inspect_history(self, n: int = 1):
        _inspect_history(self.history, n)


# if __name__ == "__main__":
#     lm = LocalLLM("llama3-8b-instruct")
#     dspy.configure(lm=lm)

#     qa = dspy.ChainOfThought("question->answer")
#     answer = qa(question="What is the capital of France?")
#     lm.inspect_history(n=1)