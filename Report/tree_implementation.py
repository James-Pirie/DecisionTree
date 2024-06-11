import pandas as pd


def decision_tree_implementation(mushroom: pd.DataFrame):
    if mushroom['odor'] == 'f':
        return "p"
    if mushroom['odor'] == 'n':
        return "e"
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
