import json


if __name__ == "__main__":
    kc_sol_annotated_path = "/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/moocradar-C_746997/questions_translated_kc_sol_annotated_mapped.json"
    kc_questions_map_path = "/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/moocradar-C_746997/kc_questions_map.json"
    with open(kc_sol_annotated_path, 'r', encoding='utf-8') as file:
        annotated_data = json.load(file)

    kc_questions_map = {}
    for q_id, q_value in annotated_data.items():
        concepts = q_value["knowledge_concepts_list"]
        for i, concept in enumerate(concepts):
            concept = concept.strip()
            annotated_data[q_id]["knowledge_concepts_list"][i] = concept
            if concept not in kc_questions_map:
                kc_questions_map[concept] = []
            kc_questions_map[concept].append(int(q_id))

    print(f"num kc: {len(kc_questions_map)}")
    with open(kc_questions_map_path, 'w', encoding='utf-8') as temp_file:
        json.dump(kc_questions_map, temp_file, ensure_ascii=False, indent=2)
    with open(kc_sol_annotated_path, 'w', encoding='utf-8') as temp_file:
        json.dump(annotated_data, temp_file, ensure_ascii=False, indent=2)