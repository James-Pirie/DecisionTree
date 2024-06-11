# 361 Assignment 1 - James Pirie

### Task 1 Code:


```python
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
```

## Task 1 Part A & B


```python
column_names = ['poisonous', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
                'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root',
                'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring',
                'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type',
                'spore-print-color',
                'population', 'habitat']

data = read_data('mushroom/agaricus-lepiota.data', column_names)
split_index = int(len(data) * 0.8)
training_data = data.iloc[:split_index]  # 80% of data for training
testing_data = data.iloc[split_index:]  # 20% of data for testing
```

### Depth 2


```python
print(train(training_data, 2))
```

    import pandas as pd
    
    
    def decision_tree_implementation(mushroom: pd.DataFrame):
        if mushroom['odor'] == 'f':
            return "p"
        if mushroom['odor'] == 'n':
            if mushroom['spore-print-color'] == 'n':
                return "e"
            if mushroom['spore-print-color'] == 'k':
                return "e"
            if mushroom['spore-print-color'] == 'w':
                return "e"
            if mushroom['spore-print-color'] == 'r':
                return "p"
        if mushroom['odor'] == 'a':
            return "e"
        if mushroom['odor'] == 'l':
            return "e"
        if mushroom['odor'] == 'c':
            return "p"
        if mushroom['odor'] == 'p':
            return "p"
        if mushroom['odor'] == 'm':
            return "p"
    


### Depth 3


```python
print(train(training_data, 3))
```

    import pandas as pd
    
    
    def decision_tree_implementation(mushroom: pd.DataFrame):
        if mushroom['odor'] == 'f':
            return "p"
        if mushroom['odor'] == 'n':
            if mushroom['spore-print-color'] == 'n':
                return "e"
            if mushroom['spore-print-color'] == 'k':
                return "e"
            if mushroom['spore-print-color'] == 'w':
                if mushroom['cap-color'] == 'n':
                    return "e"
                if mushroom['cap-color'] == 'c':
                    return "e"
                if mushroom['cap-color'] == 'y':
                    return "p"
                if mushroom['cap-color'] == 'w':
                    return "p"
                if mushroom['cap-color'] == 'p':
                    return "e"
                if mushroom['cap-color'] == 'g':
                    return "e"
            if mushroom['spore-print-color'] == 'r':
                return "p"
        if mushroom['odor'] == 'a':
            return "e"
        if mushroom['odor'] == 'l':
            return "e"
        if mushroom['odor'] == 'c':
            return "p"
        if mushroom['odor'] == 'p':
            return "p"
        if mushroom['odor'] == 'm':
            return "p"
    


### Depth 4


```python
print(train(training_data, 4))
```

    import pandas as pd
    
    
    def decision_tree_implementation(mushroom: pd.DataFrame):
        if mushroom['odor'] == 'f':
            return "p"
        if mushroom['odor'] == 'n':
            if mushroom['spore-print-color'] == 'n':
                return "e"
            if mushroom['spore-print-color'] == 'k':
                return "e"
            if mushroom['spore-print-color'] == 'w':
                if mushroom['cap-color'] == 'n':
                    return "e"
                if mushroom['cap-color'] == 'c':
                    return "e"
                if mushroom['cap-color'] == 'y':
                    return "p"
                if mushroom['cap-color'] == 'w':
                    return "p"
                if mushroom['cap-color'] == 'p':
                    return "e"
                if mushroom['cap-color'] == 'g':
                    return "e"
            if mushroom['spore-print-color'] == 'r':
                return "p"
        if mushroom['odor'] == 'a':
            return "e"
        if mushroom['odor'] == 'l':
            return "e"
        if mushroom['odor'] == 'c':
            return "p"
        if mushroom['odor'] == 'p':
            return "p"
        if mushroom['odor'] == 'm':
            return "p"
    


### Task 1 A Paragraph
The train method takes in a dataframe, and outputs a string of python code representing the decision tree. The algorithm implemented is as described in lectures, calculates which feature has the largest information gain, then splits the dataset at said feature for every potential value in the column. This repreats recursivley until the entropy of the dataframe after a split is zero, or as will be used in part B, until the stopping_depth is reached. When stopping the majority class of the data frame is returned, either p or e. This is shown, as will be explaiend in task B Paragrah, in the code snipet labeld Depth 4.

### Task 1 B Paragraph
For task B, I have implemented the stopping_depth which is taken in as an argument to the train method. The depth is tracked every time the recursion depth increases, and stops when depth reaches stopping depth. Above I display 3 cases, one where depth is 2, second where it is 3 and finaly 4. When depth is set to two the tree only splits on odor and then spore-print-color, before being forced to stop, as we are now at depth 2. For depth 3 it splits one more time at cap-color, before again being forced to stop. Finaly, at depth 4, the tree is the same at depth 3, as this time maximum entropy has been reached in all of the leaf dataframes, and no more splitting is neccesary.

## Task 1 Part C & D


```python
k_fold_cross_validation(k=10, data_input=data, depth=1)
```

    ======================================= Test 1 =======================================
    Model Test: Test Accuracy: 98.05%, Precision: 1.0, Recall: 0.95, F-Score: 0.97
    Baseline Test: Test Accuracy: 64.42%, Precision: undefined, Recall: 0.0, F-Score: undefined
    ======================================= Test 2 =======================================
    Model Test: Test Accuracy: 98.41%, Precision: 1.0, Recall: 0.96, F-Score: 0.98
    Baseline Test: Test Accuracy: 62.48%, Precision: undefined, Recall: 0.0, F-Score: undefined
    ======================================= Test 3 =======================================
    Model Test: Test Accuracy: 98.94%, Precision: 1.0, Recall: 0.97, F-Score: 0.98
    Baseline Test: Test Accuracy: 61.42%, Precision: undefined, Recall: 0.0, F-Score: undefined
    ======================================= Test 4 =======================================
    Model Test: Test Accuracy: 97.7%, Precision: 1.0, Recall: 0.94, F-Score: 0.97
    Baseline Test: Test Accuracy: 59.47%, Precision: undefined, Recall: 0.0, F-Score: undefined
    ======================================= Test 5 =======================================
    Model Test: Test Accuracy: 98.05%, Precision: 1.0, Recall: 0.95, F-Score: 0.97
    Baseline Test: Test Accuracy: 62.94%, Precision: undefined, Recall: 0.0, F-Score: undefined
    ======================================= Test 6 =======================================
    Model Test: Test Accuracy: 98.58%, Precision: 1.0, Recall: 0.96, F-Score: 0.98
    Baseline Test: Test Accuracy: 62.41%, Precision: undefined, Recall: 0.0, F-Score: undefined
    ======================================= Test 7 =======================================
    Model Test: Test Accuracy: 98.05%, Precision: 1.0, Recall: 0.95, F-Score: 0.97
    Baseline Test: Test Accuracy: 60.99%, Precision: undefined, Recall: 0.0, F-Score: undefined
    ======================================= Test 8 =======================================
    Model Test: Test Accuracy: 99.11%, Precision: 1.0, Recall: 0.98, F-Score: 0.99
    Baseline Test: Test Accuracy: 60.64%, Precision: undefined, Recall: 0.0, F-Score: undefined
    ======================================= Test 9 =======================================
    Model Test: Test Accuracy: 98.76%, Precision: 1.0, Recall: 0.97, F-Score: 0.98
    Baseline Test: Test Accuracy: 60.11%, Precision: undefined, Recall: 0.0, F-Score: undefined
    ======================================= Test 10 =======================================
    Model Test: Test Accuracy: 98.76%, Precision: 1.0, Recall: 0.97, F-Score: 0.98
    Baseline Test: Test Accuracy: 63.12%, Precision: undefined, Recall: 0.0, F-Score: undefined


### Task 1 Part C Paragraph
To test the tree, I implemented a k fold cross validation function. The function takes in the data, and splits it into k equal sized sets. It then loops throguh every set, and each time the current nth partition is used as the testing data while the rest are used for training. This produces k trees trained on slightly different data. I then calculate the accuracy, precision and recall. The above results is tested at k=10, producing 10 trees trained and tested with different variations of the dataset, the stopping_depth is set to 2. As we can see precision is perfect, accuracy is around 98 percent for every tree and recall between 0.95 and 0.99.
### Task 1 Part D Paragraph
To evaluate the output, inside the k fold cross validation I calculated the f score and then compared the test results to the baseline model. The baseline model simply predicts the majority class of the entire dataset. As we can see the trained tree is significantly more accurate than the baseline, at around 98% opposed to around 60% respectivley. I also calculated the F score, which can be used to evaluate a model. Consistently the f-score is near 0.98, which is very close to 1. This indicated the model is very succesful.

## Task 2
### Task 2 Paragraph A
One interesting modification I could do immediatly to the splitting criteria would be to split on the features with the lowest information gain. Although interesting, this would not be a suitable method to practically train a decision tree. By splitting on the lowest information gain we are barely changing the entropy of our dataset, meaning entropy will always be the highest it could be in a split. High entropy means that any given partition on said tree would be unlikley to separate edible from poisonous. And therefore will make poor predicitons. However, as long as the information gain is not 0 for every split, eventualy, after a significant amount of splits. The tree may develop some degree of accuracy. 

### Task 3 Paragraph A
My model activley works to prevent overfitting. Firstly, it evaluates and tests using 10 fold cross validation, which trains 10 different models, and keeps testing and training data separate. By segregating the data the tree has less opportunity to adapt to the noise of the training set. It should adapt to the trends of the data, not the points. And by doing this 10 times, it ensures that it is simply not overfitting just by chance, where it might once accidentaly make a split that happens to work well with the data set. Instead we can gauge the fit of the model by averging it out across the 10 cross validations.

### Additional Information
If you wish to replicate my program outside of jupiter notebook, you will need to create a python file in the same directory called baseline_model.py and use the following code:


```python
import pandas as pd


def decision_tree_implementation(mushroom: pd.DataFrame):
    """Tree implementation"""
    return "e"

```

To train, test and validate a model, you will need to run the following, which will create a new file called tree_implememntation.py, and write the newly trained trees to that, before validating.


```python
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
```

### Requirements.txt


```python
numpy
pandas
ucimlrepo
```
