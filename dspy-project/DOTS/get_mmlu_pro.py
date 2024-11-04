import datasets
import dspy


def form_options(options: list):
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str


def get_dspy_data(data_path, category_selected):
    data = datasets.load_dataset(data_path)

    val_data_dict = data["validation"][:]
    val_data = []
    for question, options, answer, cot_content, category in zip(
            val_data_dict["question"], val_data_dict["options"], val_data_dict["answer"], val_data_dict["cot_content"],
            val_data_dict["category"]
    ):
        if category == category_selected:
            question_content = 'Q: ' + question + '\n' + form_options(options) + '\n'
            val_data.append(dspy.Example(question=question_content, answer=answer, reasoning=cot_content).
                            with_inputs("question", "reasoning"))

    test_data_dict = data["test"][:]
    test_data = []
    for question, options, answer, category in zip(
        test_data_dict["question"], test_data_dict["options"], test_data_dict["answer"], test_data_dict["category"]
    ):
        if category == category_selected:
            question_content = 'Q: ' + question + '\n' + form_options(options) + '\n'
            test_data.append(dspy.Example(question=question_content, answer=answer).with_inputs("question"))

    return val_data, test_data
