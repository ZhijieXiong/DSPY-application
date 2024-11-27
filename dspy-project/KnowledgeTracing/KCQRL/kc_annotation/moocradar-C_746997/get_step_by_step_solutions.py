import json
from openai import OpenAI
import os
import argparse
from tqdm import tqdm

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process some files.')
parser.add_argument('--original_question_file', type=str,
                    default="/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/moocradar-C_746997/question_translated.json",
                    help='Path to the original question file')
parser.add_argument('--annotated_question_file', type=str,
                    default="/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/moocradar-C_746997/questions_translated_step_by_step.json",
                    help='Path to the target question file')
args = parser.parse_args()

original_question_file = args.original_question_file
annotated_question_file = args.annotated_question_file

client = OpenAI(
    api_key=os.getenv("BAILIAN_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

system_prompt = """Your task is to generate the clear and concise step by step solutions of the provided problem. Please consider the below instructions in your generation. 

- You will be provided with the final answer.  
- It is important that your generated step by step solution should be understandable as stand-alone, meaning that the student should not need to additionally check final answer or explanation provided. 
- Please provide your step-by-step solution as each step in a new line. Don't enumerate the steps. Don't put any bullet point. Separate the solution steps only with `####`. In short, I want to be able to parse your listed solution steps via splitting your output with `####` 
- Don't generate any text other than the step by step solution described earlier.
- Make your step-by-step solution concise as described earlier."""

user_prompt_template = """Question: {}
Final Answer: {}
Step by Step Solution: """


def get_soluton_steps(item):
    # Get the full user prompt
    full_user_prompt = user_prompt_template.format(item['question'], item["answer"])

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
        max_tokens=512,
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
num2evaluate = 550
progress_bar = tqdm(total=num2evaluate)
counter = 0
for key, item in data.items():
    if counter >= num2evaluate:
        break

    if key in annotated_question.keys():
        continue

    try:
        item['step_by_step_solution_text'] = get_soluton_steps(item)
        annotated_question[key] = item
    except:
        break

    counter += 1
    progress_bar.update(1)

with open(annotated_question_file, 'w', encoding='utf-8') as temp_file:
    json.dump(annotated_question, temp_file, ensure_ascii=False, indent=2)
