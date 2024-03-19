import pandas as pd
import math


def read_data(filename: str, features: str):
    """Creating a DataFrame from a dictionary"""
    df = pd.read_csv(filename, header=None, names=features)
    df = df[~df.isin(['?']).any(axis=1)]
    df = df.sample(frac=1, random_state=22).reset_index(drop=True)  # Shuffle the DataFrame with the provided seed
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


def information_gain(data_input: pd.DataFrame, value_to_split: str):
    """Calculate the information gain for every value of every feature in a DataFrame"""
    # entropy of entire column
    entropy_before_split = entropy(data_input)
    total_size = data_input.shape[0]  # number of total rows

    # split the data based on the value to split the column
    data_split_on_value = data_input[data_input.iloc[:, 1] == value_to_split]
    data_split_on_value_size = data_split_on_value.shape[0]  # get size of new DataFrame

    data_split_without_value = data_input[data_input.iloc[:, 1] != value_to_split]  # get the rest of the data set
    data_split_without_value_size = data_split_without_value.shape[0]  # size of rest of data set

    # calculate and return information gain
    return entropy_before_split - \
           ((data_split_on_value_size / total_size) * entropy(data_split_on_value)) - \
           ((data_split_without_value_size / total_size) * entropy(data_split_without_value))


def find_split(data_input: pd.DataFrame):
    """Find the largest entropy in a DataFrame"""
    poisonous = data_input.iloc[:, 0]  # get the values from the first column (poisonous column)
    features = data_input.iloc[:, 1:]  # remove classifier value from the list
    information_gains = []

    # iterate through every column
    for feature in features:
        current_table = data_input.loc[:, [poisonous.name, feature]]
        all_possible_values = data_input[feature].unique()

        # iterate through every potential value in a column
        for value in all_possible_values:
            current_information_gain = {"feature": feature,
                                        "value": value,
                                        "information_gain": information_gain(current_table, value)}
            information_gains.append(current_information_gain)

    # return the largest information gain
    return [max(information_gains, key=lambda x: x['information_gain'])]


def split_data(data_input: pd.DataFrame, largest_information_gain: dict):
    """Split a DataFrame into two based upon an information gain score"""
    split_with_value = data_input[data_input[largest_information_gain['feature']] == largest_information_gain['value']]
    split_without_value = data_input[data_input[largest_information_gain['feature']] != largest_information_gain['value']]
    return split_with_value, split_without_value


def greed_recursive_splitting(data_input: pd.DataFrame, max_depth: int, current_depth, splitting_point: dict):
    """Recursively split the DataFrame until either entropy is zero or depth limit has been reached """
    tab = "    "
    print(f"{current_depth * tab}if mushroom['{splitting_point['feature']}'] == '{splitting_point['value']}':")

    current_depth += 1

    if current_depth <= max_depth and entropy(data_input) != 0:
        split_frames = split_data(data_input, splitting_point)

        greed_recursive_splitting(split_frames[0], max_depth, current_depth, find_split(split_frames[0])[0])
        greed_recursive_splitting(split_frames[1], max_depth, current_depth, find_split(split_frames[1])[0])

    else:
        print(f'{(current_depth + 1) * tab}return "p"')


def train(data_input: pd.DataFrame, depth: int):
    """Take in a DataFrame and print out the nodes for the decision tree"""
    splitting_point = find_split(data_input)[0]
    greed_recursive_splitting(data_input, depth, 0, splitting_point)
    print("return 'e'")


def decision_tree_implementation(mushroom: pd.DataFrame):
    if mushroom['odor'] == 'f':
        if mushroom['cap-shape'] == 'x':
            return "p"
        if mushroom['gill-size'] == 'b':
            if mushroom['ring-number'] == 'o':
                if mushroom['cap-shape'] == 'f':
                    return "p"
                if mushroom['habitat'] == 'p':
                    if mushroom['cap-shape'] == 'x':
                        return "p"
                    if mushroom['stalk-surface-above-ring'] == 'y':
                        if mushroom['cap-shape'] == 'b':
                            return "p"
                        if mushroom['cap-shape'] == 'b':
                            return "p"
            if mushroom['odor'] == 'n':
                if mushroom['population'] == 'c':
                    if mushroom['cap-shape'] == 'f':
                        return "p"
                    if mushroom['cap-shape'] == 'x':
                        return "p"
                if mushroom['stalk-shape'] == 'e':
                    if mushroom['cap-shape'] == 'x':
                        return "p"
                    if mushroom['cap-shape'] == 'f':
                        return "p"
    return 'e'


def test(test_data: pd.DataFrame):
    correct = 0
    for i in range(testing_data.shape[0]):
        if test_data.iloc[i]['poisonous'] == decision_tree_implementation(test_data.iloc[i]):
            correct += 1
    print(f"{(correct / testing_data.shape[0]) * 100}%")


if __name__ == '__main__':
    column_names = ['poisonous', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                    'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                    'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
                    'spore-print-color',
                    'population', 'habitat']

    data = read_data('mushroom/agaricus-lepiota.data', column_names)
    split_index = int(len(data) * 0.8)

    training_data = data.iloc[:split_index]
    testing_data = data.iloc[split_index:]

    train(training_data, 100)
    test(testing_data)
