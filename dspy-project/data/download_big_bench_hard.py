import os
import inspect
from datasets import load_dataset


if __name__ == "__main__":
    all_configs = [
        'tracking_shuffled_objects_seven_objects',
        'salient_translation_error_detection',
        'tracking_shuffled_objects_three_objects',
        'geometric_shapes',
        'object_counting',
        'word_sorting',
        'logical_deduction_five_objects',
        'hyperbaton',
        'sports_understanding',
        'logical_deduction_seven_objects',
        'multistep_arithmetic_two',
        'ruin_names',
        'causal_judgement',
        'logical_deduction_three_objects',
        'formal_fallacies',
        'snarks',
        'boolean_expressions',
        'reasoning_about_colored_objects',
        'dyck_languages',
        'navigate',
        'disambiguation_qa',
        'temporal_sequences',
        'web_of_lies',
        'tracking_shuffled_objects_five_objects',
        'penguins_in_a_table',
        'movie_recommendation',
        'date_understanding'
    ]

    current_file_name = inspect.getfile(inspect.currentframe())
    current_dir = os.path.dirname(current_file_name)
    for cat in all_configs:
        sub_dir = os.path.join(current_dir, "BIG-BENCH-HARD", cat)
        if not os.path.exists(sub_dir):
            os.mkdir(sub_dir)
        ds = load_dataset("maveriq/bigbenchhard", cat)
        ds.save_to_disk(sub_dir)
