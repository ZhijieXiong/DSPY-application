import json
import os
from tqdm import tqdm


def evaluate_from_last(data, output_path, dspy_predictor, args):
    if not os.path.exists(output_path):
        qa_results = {}
    else:
        with open(output_path, 'r') as file:
            qa_results = json.load(file)

    indices_evaluated = list(map(int, qa_results.keys()))
    idx_start = 0
    if len(indices_evaluated) == 0:
        data2evaluate = data[:args.num2evaluate]
    elif len(qa_results) >= len(data):
        return qa_results
    else:
        idx_start = max(indices_evaluated) + 1
        data2evaluate = data[idx_start: idx_start + args.num2evaluate]

    for i, example in enumerate(tqdm(data2evaluate)):
        q_id = i + idx_start

        try:
            predict_response = dspy_predictor(question=example.question)
            qa_results[q_id] = {
                "question": example.question,
                "answer": example.answer,
                "response": predict_response.answer
            }
            if args.prompt_method in ["CoT", "PoT"]:
                qa_results[q_id]["reasoning"] = predict_response.reasoning
        except:
            # 可能LLM不按照格式输出，导致解析错误，目前就随便输出错误字符串，当作做错
            qa_results[q_id] = {
                "question": example.question,
                "answer": example.answer,
                "response": "Error parsing field answer"
            }
            if args.prompt_method in ["CoT", "PoT"]:
                qa_results[q_id]["reasoning"] = "Error parsing field reasoning"

    return qa_results
