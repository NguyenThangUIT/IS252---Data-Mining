import numpy as np

def calculate_entropy(data, target_attribute):
    total_samples = len(data)
    entropy = 0
    for class_value in data[target_attribute].unique():
        p_class = len(data[data[target_attribute] == class_value]) / total_samples
        if p_class > 0:
            entropy -= p_class * np.log2(p_class)
    return entropy

def calculate_gain(data, attribute, target_attribute):
    total_entropy = calculate_entropy(data, target_attribute)
    weighted_entropy = 0
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        weighted_entropy += (len(subset) / len(data)) * calculate_entropy(subset, target_attribute)
    return total_entropy - weighted_entropy

def gain_id3_decision_tree(data, target_attribute, attributes=None):
    if attributes is None:
        attributes = list(data.columns.drop(target_attribute))

    if len(data[target_attribute].unique()) == 1:
        return data[target_attribute].unique()[0]

    if not attributes:
        return data[target_attribute].mode()[0]

    best_attribute = max(
        attributes,
        key=lambda attr: calculate_gain(data, attr, target_attribute)
    )

    tree = {best_attribute: {}}
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        new_attrs = [attr for attr in attributes if attr != best_attribute]
        tree[best_attribute][value] = gain_id3_decision_tree(subset, target_attribute, new_attrs)

    return tree

def calculate_gini(data, attribute, target_attribute):
    total_samples = len(data)
    weighted_gini = 0
    for value in data[attribute].unique():
        subset = data[data[attribute] == value]
        subset_size = len(subset)
        gini = 1
        for class_value in subset[target_attribute].unique():
            p = len(subset[subset[target_attribute] == class_value]) / subset_size
            gini -= p ** 2
        weighted_gini += (subset_size / total_samples) * gini
    return weighted_gini

def gini_id3_decision_tree(data, target_attribute, attributes=None):
    if attributes is None:
        attributes = list(data.columns.drop(target_attribute))

    if len(data[target_attribute].unique()) == 1:
        return data[target_attribute].unique()[0]

    if not attributes:
        return data[target_attribute].mode()[0]

    best_attribute = min(
        attributes,
        key=lambda attr: calculate_gini(data, attr, target_attribute)
    )

    tree = {best_attribute: {}}
    for value in data[best_attribute].unique():
        subset = data[data[best_attribute] == value]
        new_attrs = [attr for attr in attributes if attr != best_attribute]
        tree[best_attribute][value] = gini_id3_decision_tree(subset, target_attribute, new_attrs)

    return tree

def extract_rules(tree, target_attribute, current_rule=None, rules=None):
    if rules is None:
        rules = []
    if current_rule is None:
        current_rule = []

    if not isinstance(tree, dict):
        rule = "IF " + " AND ".join(current_rule) + f" THEN {target_attribute} = {tree}"
        rules.append(rule)
        return rules

    attribute = next(iter(tree))
    for value, subtree in tree[attribute].items():
        condition = f"{attribute} = {value}"
        extract_rules(subtree, target_attribute, current_rule + [condition], rules)

    return rules