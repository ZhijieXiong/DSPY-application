import json
import os
from openai import OpenAI
from tqdm import tqdm
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--original_question_file', type=str,
                    default="/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/edi2020-task34/questions_step_by_step.json",
                    help='Path to the output question file of get_step_by_step_solutions.py')
parser.add_argument('--annotated_question_file', type=str,
                    default="/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/edi2020-task34/questions_kc_annotation.json",
                    help='Path to the target question file')
args = parser.parse_args()

original_question_file = args.original_question_file
annotated_question_file = args.annotated_question_file

client = OpenAI(
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

system_prompt = """You will be provided with a question, its final answer and its step by step solution (each step is separated by `####`). Your task is to provide the concise and comprehensive list of knowledge concepts required to correctly answer the questions.

Please carefully follow the below instructions: 
- Provide multiple knowledge concepts only when it is actually needed. 
- Some questions require a figure, which you won't be provided. As the step-by-step solution is already provided, Use your judgement to infer which knowledge concept(s) might be needed.
- For a small set of solutions, their last step(s) might be missing due to limited token size. Use your judgement based on your input and your ability to infer how the solution would conclude. 
- If annotated step-by-step solution involves more advanced techniques, use your judgment for more simplified alternatives.
- IMPORTANT: Provide only  the knowledge concepts, but nothing else. Separate them with `####`, and please don't use any enumeration or bullet points. In short, I want to be able to parse your listed knowledge concepts via splitting your output with `####`."""

user_prompt_template = """Question: {}
Final Answer: {}
Step by Step Solution: {}"""


def get_kcs(item):
    # Get the full user prompt
    full_user_prompt = user_prompt_template.format(item['question'], item["answer"], item["step_by_step_solution_text"])

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": full_user_prompt
            }
        ],
        temperature=0,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content  # Assuming this returns the converted text


# Load JSON data
with open(original_question_file, 'r', encoding='utf-8') as file:
    data = json.load(file)

if not os.path.exists(annotated_question_file):
    annotated_question = {}
else:
    with open(annotated_question_file, 'r') as file:
        annotated_question = json.load(file)
num2evaluate = 450
progress_bar = tqdm(total=num2evaluate)
counter = 0
for key, item in data.items():
    if counter >= num2evaluate:
        break

    if key in annotated_question.keys():
        continue

    try:
        item['knowledge_concepts_text'] = get_kcs(item)
        annotated_question[key] = item
    except:
        break

    counter += 1
    progress_bar.update(1)

with open(annotated_question_file, 'w', encoding='utf-8') as temp_file:
    json.dump(annotated_question, temp_file, ensure_ascii=False, indent=2)
