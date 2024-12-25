import json
import os
import time
from tqdm import tqdm
from zhipuai import ZhipuAI


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def write_json(json_data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False)


if __name__ == "__main__":
    client = ZhipuAI(api_key=os.getenv("GLM_API_KEY"))
    question_path = "/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/edi2020-task34/question.json"
    emb_path = "/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/KCQRL/que_emb/edi2020-task34/qid2content_emb.json"
    num2cal = 948
    dim_emb = 1024

    question = load_json(question_path)
    if os.path.exists(emb_path):
        emb = load_json(emb_path)
    else:
        emb = {}

    i = 0
    process_bar = tqdm(total=num2cal)
    for q_id, q in question.items():
        if i >= num2cal:
            break
        if q_id in emb:
            continue

        if i % 10 == 0:
            time.sleep(1)

        try:
            q_text = q["content"]
            response = client.embeddings.create(
                model="embedding-3",
                input=[
                    q_text,
                ],
                extra_body={
                    "dimensions": dim_emb
                }
            )
            emb[q_id] = response.data[0].embedding
            i += 1
            process_bar.update(1)
        except:
            break

    write_json(emb, emb_path)
