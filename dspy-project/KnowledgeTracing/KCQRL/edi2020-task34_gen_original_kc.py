import json
import time
from tqdm import tqdm
import pandas as pd


if __name__ == "__main__":
    c_id_map_path = "/Users/dream/myProjects/dlkt-release/lab/dataset_preprocessed/edi2020-task34/concept_id_map_single_concept.csv"
    c_id_name_path = "/Users/dream/myProjects/dlkt-release/lab/dataset_raw/edi2020/metadata/subject_metadata.csv"
    kc_meta_path = "/Users/dream/myProjects/dlkt-release/lab/dataset_preprocessed/edi2020-task34/concept_id_map_single_concept.csv"
    json_file_dataset_path = "/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/edi2020-task34/questions_translated_kc_sol_annotated_mapped.json"
    questions_translated_path = "/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/edi2020-task34/questions_translated_kc_sol_annotated_mapped_original_kc.json"
    kc_clusters_path = "/Users/dream/myProjects/DSPY_research/dspy-project/KnowledgeTracing/data/edi2020-task34/kc_no_clusters_original_kc.json"

    c_id_map = pd.read_csv(c_id_map_path)
    c_id_name = pd.read_csv(c_id_name_path)
    merged_df = pd.merge(c_id_map, c_id_name, left_on='concept_id', right_on='SubjectId', how='inner')
    kc_data = dict(zip(merged_df['concept_mapped_id'], merged_df['Name']))
    kc_clusters_hdbscan = {}
    for c_id, kc in tqdm(kc_data.items()):
        kc_clusters_hdbscan[c_id] = [kc.strip()]

    with open(json_file_dataset_path, 'r') as f:
        data = json.load(f)
    for q_id, q in tqdm(data.items()):
        data[q_id]["mapping_step_kc_gpt-4o"] = ", ".join([f"{step + 1}-1" for step in range(len(q["step_by_step_solution_list"]))])
        kcs = []
        for kc in q['concepts']:
            kcs.append(kc)
            data[q_id]["knowledge_concepts_list"] = kcs
            data[q_id]["knowledge_concepts_text"] = "####".join(kcs)

    with open(questions_translated_path, 'w') as f:
        json.dump(data, f)

    with open(kc_clusters_path, 'w') as f:
        json.dump(kc_clusters_hdbscan, f)
