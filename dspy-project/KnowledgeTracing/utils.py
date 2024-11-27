import os
import json


def read_preprocessed_file(data_path):
    assert os.path.exists(data_path), f"{data_path} not exist"
    with open(data_path, "r") as f:
        all_lines = f.readlines()
        first_line = all_lines[0].strip()
        seq_interaction_keys_str = first_line.split(";")
        id_keys_str = seq_interaction_keys_str[0].strip()
        seq_keys_str = seq_interaction_keys_str[1].strip()
        id_keys = id_keys_str.split(",")
        seq_keys = seq_keys_str.split(",")
        keys = id_keys + seq_keys
        num_key = len(keys)
        all_lines = all_lines[1:]
        data = []
        for i, line_str in enumerate(all_lines):
            if i % num_key == 0:
                item_data = {}
            current_key = keys[int(i % num_key)]
            if current_key in ["time_factor_seq", "hint_factor_seq", "attempt_factor_seq", "correct_float_seq"]:
                line_content = list(map(float, line_str.strip().split(",")))
            else:
                line_content = list(map(int, line_str.strip().split(",")))
            if len(line_content) == 1:
                # 说明是序列级别的特征，即user id、seq len、segment index等等
                item_data[current_key] = line_content[0]
            else:
                # 说明是interaction级别的特征，即question id等等
                item_data[current_key] = line_content
            if i % num_key == (num_key - 1):
                data.append(item_data)

    return data


def write2file(data, data_path):
    # id_keys表示序列级别的特征，如user_id, seq_len
    # seq_keys表示交互级别的特征，如question_id, concept_id
    id_keys = []
    seq_keys = []
    for key in data[0].keys():
        if type(data[0][key]) == list:
            seq_keys.append(key)
        else:
            id_keys.append(key)

    # 不知道为什么，有的数据集到这的时候，数据变成float类型了（比如junyi2015，如果预处理部分数据，就是int，但是如果全量数据，就是float）
    id_keys_ = set(id_keys).intersection({"user_id", "school_id", "premium_pupil", "gender", "seq_len", "campus",
                                          "dataset_type", "order"})
    seq_keys_ = set(seq_keys).intersection({"question_seq", "concept_seq", "correct_seq", "time_seq", "use_time_seq",
                                            "use_time_first_seq", "num_hint_seq", "num_attempt_seq", "age_seq",
                                            "question_mode_seq"})
    for item_data in data:
        for k in id_keys_:
            try:
                item_data[k] = int(item_data[k])
            except ValueError:
                print(f"value of {k} has nan")
        for k in seq_keys_:
            try:
                item_data[k] = list(map(int, item_data[k]))
            except ValueError:
                print(f"value of {k} has nan")

    with open(data_path, "w") as f:
        first_line = ",".join(id_keys) + ";" + ",".join(seq_keys) + "\n"
        f.write(first_line)
        for item_data in data:
            for k in id_keys:
                f.write(f"{item_data[k]}\n")
            for k in seq_keys:
                f.write(",".join(map(str, item_data[k])) + "\n")


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        result = json.load(f)
    return result


def write_json(json_data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)