import json
import dspy


def read_data(data_path):
	data = []
	with open(data_path, "r") as f:
		for line in f:
			item = json.loads(line)
			reasoning, answer = item["answer"].split("####")
			data.append(dspy.Example(question=item["question"], answer=answer, reasoning=reasoning).with_inputs("question"))
	return data


if __name__ == "__main__":
	train_data = read_data("/dspy/gsm8k_data/train.jsonl")
	print(train_data[0])
