import pandas as pd
import math
import numpy as np
from baseline_model import decision_tree_implementation


import pandas as pd
import math
import numpy as np
from baseline_model import decision_tree_implementation


def read_data(filename: str, features: str):
    """Creating a DataFrame from a dictionary"""
    df = pd.read_csv(filename, header=None, names=features)
    df = df[~df.isin(['?']).any(axis=1)]
    df = df.sample(frac=1, random_state=4).reset_index(drop=True)  # Shuffle the DataFrame with the provided seed
    return df


def entropy(data_input: pd.DataFrame):
    """Calculate the entropy of a DataFrame"""
    total_rows = data_input.shape[0]
    is_poisonous = data_input[data_input.iloc[:, 0] == 'p'].shape[0]
    is_not_poisonous = data_input[data_input.iloc[:, 0] == 'e'].shape[0]

    if is_poisonous == 0 or is_not_poisonous == 0:
        return 0

    if is_poisonous == is_not_poisonous:
        return 1

    # equation for entropy
    return - (is_poisonous / total_rows) * math.log2(is_poisonous / total_rows)\
           - (is_not_poisonous / total_rows) * math.log2(is_not_poisonous / total_rows)


def information_gain(data_input: pd.DataFrame):
    """Calculate the information gain for every value of a feature in a DataFrame"""
    # entropy of entire column
    entropy_before_split = entropy(data_input)
    total_size = data_input.shape[0]  # number of total rows

    all_possible_values = data_input.iloc[:, 1].unique().tolist()
    weighted_entropy_sum = 0

    for value in all_possible_values:
        # split the data based on the value to split the column
        data_split_on_value = data_input[data_input.iloc[:, 1] == value]
        data_split_on_value_size = data_split_on_value.shape[0]  # get size of new DataFrame

        weighted_entropy_sum += (data_split_on_value_size / total_size) * entropy(data_split_on_value)

    # calculate and return information gain
    return entropy_before_split - weighted_entropy_sum


def find_split(data_input: pd.DataFrame):
    """Find the largest information-gain in a DataFrame"""
    poisonous = data_input.iloc[:, 0]  # get the values from the first column (poisonous column)
    features = data_input.iloc[:, 1:]  # remove classifier value from the list
    information_gains = []

    # iterate through every column
    for feature in features:
        current_table = data_input.loc[:, [poisonous.name, feature]]
        current_information_gain = {"feature": feature,
                                    "information_gain": information_gain(current_table)}
        information_gains.append(current_information_gain)

    # return the largest information gain
    return max(information_gains, key=lambda x: x['information_gain'])


def split_data(data_input: pd.DataFrame, largest_information_gain: dict):
    """Split a DataFrame based upon an information gain score"""
    all_possible_values = data_input[largest_information_gain['feature']].unique().tolist()
    split_frames = []
    for value in all_possible_values:
        split_with_value = data_input[data_input[largest_information_gain['feature']] == value]
        split_frames.append(split_with_value)

    return split_frames


def get_majority(data_input: pd.DataFrame):
    return data_input.iloc[:, 0].value_counts().idxmax()


def greedy_recursive_splitting(data_input: pd.DataFrame, stopping_depth: int, current_depth: int) -> str:
    """Recursively split the DataFrame until either entropy is zero or depth limit has been reached """
    tab = "    "
    current_depth += 1
    splitting_point = find_split(data_input)

    all_values_to_split = data_input[splitting_point['feature']].unique().tolist()
    frame_index = 0

    output = ""

    if entropy(data_input) != 0:
        for value in all_values_to_split:
            output += f"{current_depth * tab}if mushroom['{splitting_point['feature']}'] == '{value}':\n"
            split_frames = split_data(data_input, splitting_point)

            if current_depth == stopping_depth or entropy(split_frames[frame_index]) == 0:
                output += f'{(current_depth + 1) * tab }return "{get_majority(split_frames[frame_index])}"\n'
            else:
                for frame in split_frames:
                    if current_depth <= stopping_depth:
                        output += greedy_recursive_splitting(frame, stopping_depth, current_depth)
            frame_index += 1

    return output


def train(data_input: pd.DataFrame, stopping_depth: int):
    """Take in a DataFrame and return a string of the code for the nodes for the decision tree"""
    tree_setup = \
        "import pandas as pd\n\n\ndef decision_tree_implementation(mushroom: pd.DataFrame):\n"
    return tree_setup + greedy_recursive_splitting(data_input, stopping_depth, 0)


def evaluate_model(test_data: pd.DataFrame, decision_tree_implementation):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    correct = 0

    for i in range(test_data.shape[0]):
        if test_data.iloc[i]['poisonous'] == decision_tree_implementation(test_data.iloc[i]):
            correct += 1

        if test_data.iloc[i]['poisonous'] == 'p' and decision_tree_implementation(test_data.iloc[i]) == 'p':
            true_positive += 1

        elif test_data.iloc[i]['poisonous'] == 'p' and decision_tree_implementation(test_data.iloc[i]) == 'e':
            false_negative += 1

        elif test_data.iloc[i]['poisonous'] == 'e' and decision_tree_implementation(test_data.iloc[i]) == 'e':
            true_negative += 1

        elif test_data.iloc[i]['poisonous'] == 'e' and decision_tree_implementation(test_data.iloc[i]) == 'p':
            false_positive += 1

    accuracy = (correct / test_data.shape[0]) * 100

    if true_positive + false_positive != 0:
        precision = (true_positive / (true_positive + false_positive))
        precision = round(precision, 2)
    else:
        precision = 'undefined'

    if true_positive + false_negative != 0:
        recall = (true_positive / (true_positive + false_negative))
        recall = round(recall, 2)
    else:
        recall = 'undefined'

    try:
        f_score = 2 * ((precision * recall) / (precision + recall))
        f_score = round(f_score, 2)
    except:
        f_score = 'undefined'

    accuracy = round(accuracy, 2)
    return f"Test Accuracy: {accuracy}%, Precision: {precision}, Recall: {recall}, F-Score: {f_score}"


def test(test_data: pd.DataFrame):
    # import here, in case tree has just been trained
    from tree_implementation import decision_tree_implementation
    return evaluate_model(test_data, decision_tree_implementation)


def baseline_test(test_data: pd.DataFrame):
    return evaluate_model(test_data, decision_tree_implementation)


def k_fold_cross_validation(k: int, data_input: pd.DataFrame, depth):
    """Divide the dataset into k partitions, """
    k_data_frames = np.array_split(data_input, k)

    for i in range(len(k_data_frames)):
        # use 1/kth of the data for testing
        testing_frame = pd.DataFrame(k_data_frames[i])
        # use the rest of the data for training
        training_frame = pd.concat([frame for j, frame in enumerate(k_data_frames) if j != i])
        print(f"======================================= Test {i + 1} =======================================")
        write_to_file(train(training_frame, depth))
        print(f"Model Test: {test(testing_frame)}")
        print(f"Baseline Test: {baseline_test(testing_frame)}")


def write_to_file(decision_tree: str):
    """Write the decision tree to a .py file"""
    with open('tree_implementation.py', 'w') as file:
        file.write(decision_tree)

    file.close()


if __name__ == '__main__':
    column_names = ['poisonous', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
                    'spore-print-color',
                    'population', 'habitat']

    data = read_data('mushroom/agaricus-lepiota.data', column_names)
    split_index = int(len(data) * 0.8)

    k_fold_cross_validation(10, data, 1)
