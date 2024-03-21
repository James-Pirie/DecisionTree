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

    # return the largest 3 information gains
    top_3_gains = sorted(information_gains, key=lambda x: x['information_gain'], reverse=True)
    return top_3_gains[:3]


def split_data(data_input: pd.DataFrame, features_to_split: list):
    """Split a DataFrame into two based upon an information gain score"""
    split_frames = []
    features_values = []
    for feature in features_to_split:
        features_values.append(feature['value'])
        split_with_value = data_input[data_input[feature['feature']] == feature['value']]
        split_frames.append(split_with_value)

        if len(features_to_split) == 1:
            split_without_value = data_input[data_input[feature['feature']] != feature['value']]
            split_frames.append(split_without_value)

    if len(features_to_split) > 1:
        masks = [data_input[feature['feature']] != value for feature, value in zip(features_to_split, features_values)]

        # Combine masks using logical AND operation
        combined_mask = masks[0]
        for mask in masks[1:]:
            combined_mask &= mask
        split_frames.append(data_input[combined_mask])

    return split_frames


def greedy_recursive_splitting(data_input: pd.DataFrame, max_depth: int, current_depth, splitting_points: list):
    """Recursively split the DataFrame until either entropy is zero or depth limit has been reached """
    tab = "    "

    # create root node
    print(f"{current_depth * tab}if mushroom['{splitting_points[0]['feature']}'] == '{splitting_points[0]['value']}':")
    current_depth += 1

    # if not exceeding maximum depth, and there is entropy
    if current_depth <= max_depth and entropy(data_input) != 0:

        split_frames = split_data(data_input, splitting_points)
        for frame in split_frames:
            potential_split_points = find_split(frame)
            # if not leaf node create children
            if len(potential_split_points) > 0:
                primary_split = potential_split_points[0]

                for split in potential_split_points[:1]:
                    print(split)
                    if split['feature'] == primary_split['feature'] and split['value'] != primary_split['value']:
                        greedy_recursive_splitting(frame, max_depth, current_depth, potential_split_points)

    else:
        print(f"{current_depth * tab}return 'p'")

    """
    if current_depth <= max_depth and entropy(data_input) != 0:
        split_frames = split_data(data_input, splitting_point)

        for frame in split_frames:
            potential_split_points = find_split(frame)

            if len(potential_split_points) == 0:
                print(f"{current_depth * tab}return 'p' - end")

            else:
                initial_point = potential_split_points[0]
                final_split_points = [initial_point]

                for point in potential_split_points[1:]:
                    if point['feature'] == initial_point['feature'] and point['value'] != initial_point['value']:
                        final_split_points.append(point)

                greedy_recursive_splitting(frame, max_depth, current_depth, final_split_points)

    else:
        print(f"{current_depth * tab}return 'p'")
    """


def train(data_input: pd.DataFrame, depth: int):
    """Take in a DataFrame and print out the nodes for the decision tree"""
    # find the splitting point for the root node
    splitting_point = find_split(data_input)
    greedy_recursive_splitting(data_input, depth, 0, splitting_point)

    # at the end of the tree, return not-poisonous
    print("return 'e'")


def decision_tree_implementation(mushroom: pd.DataFrame):
    if mushroom['odor'] == 'f':
        if mushroom['cap-shape'] == 'x':
            return 'p'
        if mushroom['cap-shape'] == 'x':
            return 'p'
        if mushroom['spore-print-color'] == 'r':
            if mushroom['cap-shape'] == 'f':
                return 'p'
            if mushroom['population'] == 'c':
                if mushroom['cap-shape'] == 'k':
                    return 'p'
                if mushroom['cap-shape'] == 'f':
                    return 'p'
        if mushroom['cap-shape'] == 'x':
            return 'p'
        if mushroom['gill-size'] == 'b':
            if mushroom['cap-shape'] == 'f':
                return 'p'
            if mushroom['stalk-shape'] == 't':
                if mushroom['cap-shape'] == 'f':
                    return 'p'
                if mushroom['cap-shape'] == 'x':
                    return 'p'
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

    train(training_data, 5)
    test(testing_data)
