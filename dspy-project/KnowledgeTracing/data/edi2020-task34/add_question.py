import json

with open("/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/edi2020-task34/question.json", 'r', encoding='utf-8') as file:
    data = json.load(file)

for k,v in data.items():
    v["question"] = v["content"]

with open("/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/edi2020-task34/question.json", 'w', encoding='utf-8') as temp_file:
    json.dump(data, temp_file, ensure_ascii=False, indent=2)

