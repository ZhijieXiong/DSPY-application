import json


if __name__ == "__main__":
    kc_sol_annotated_path = "/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/moocradar-C_746997/questions_translated_kc_sol_annotated_mapped.json"

    with open(kc_sol_annotated_path, 'r', encoding='utf-8') as file:
        annotated_data = json.load(file)

    # 检查
    for q_id, q_value in annotated_data.items():
        concepts = q_value["knowledge_concepts_list"]
        steps = q_value["step_by_step_solution_list"]

        try:
            concept_step_pairs = q_value["mapping_step_kc_gpt-4o"].split(",")
            c_ids = []
            s_ids = []
            for concept_step_pair in concept_step_pairs:
                concept_step_pair = concept_step_pair.strip()
                s_id, c_id = concept_step_pair.split("-")
                c_ids.append(int(c_id))
                s_ids.append(int(s_id))

            if max(c_ids) > len(concepts):
                print(f"{q_id} concepts error")

            if max(s_ids) > len(steps):
                print(f"{q_id} steps error")

        except:
            print(f"{q_id} parse error")
