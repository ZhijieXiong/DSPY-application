import dspy
import os
import datasets


def get_dspy_data(data_dir, subset):
    ds = datasets.load_from_disk(os.path.join(data_dir, subset))
    data_dict = ds["train"][:]
    data = []
    for question, answer in zip(data_dict["input"], data_dict["target"]):
        data.append(dspy.Example(question=question, answer=answer).with_inputs("question"))
    return data
