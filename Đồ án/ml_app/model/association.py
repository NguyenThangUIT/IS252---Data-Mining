import itertools
from collections import defaultdict

def calculate_support(itemset, transactions):
    count = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction)))
    return count / len(transactions)


def apriori(transactions, min_support):
    """Tìm tất cả tập phổ biến thỏa min-support"""
    item_counts = defaultdict(int)
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1

    frequent_itemsets = [[item] for item, count in item_counts.items() if count / len(transactions) >= min_support]
    k = 2
    while True:
        candidate_itemsets = []
        for i in range(len(frequent_itemsets)):
            for j in range(i + 1, len(frequent_itemsets)):
                itemset = sorted(list(set(frequent_itemsets[i]) | set(frequent_itemsets[j])))
                if len(itemset) == k and itemset not in candidate_itemsets:
                    candidate_itemsets.append(itemset)

        frequent_itemsets_k = []
        for itemset in candidate_itemsets:
            support = calculate_support(itemset, transactions)
            if support >= min_support:
                frequent_itemsets_k.append(itemset)

        if not frequent_itemsets_k:
            break

        frequent_itemsets.extend(frequent_itemsets_k)
        k += 1

    return frequent_itemsets


def find_maximal_frequent_itemsets(frequent_itemsets):
    """Tìm tập phổ biến tối đại"""
    maximal_itemsets = []
    for itemset in frequent_itemsets:
        is_maximal = True
        for other_itemset in frequent_itemsets:
            if set(itemset) < set(other_itemset):
                is_maximal = False
                break
        if is_maximal:
            maximal_itemsets.append(itemset)
    return maximal_itemsets


def generate_rules(maximal_itemsets):
    """Sinh luật từ tập phổ biến tối đại"""
    rules = []
    for itemset in maximal_itemsets:
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                consequent = list(set(itemset) - set(antecedent))
                rules.append((list(antecedent), consequent))
    return rules


def calculate_confidence(antecedent, consequent, transactions):
    antecedent_support = calculate_support(antecedent, transactions)
    combined_support = calculate_support(antecedent + consequent, transactions)
    return combined_support / antecedent_support if antecedent_support else 0


def find_satisfied_rules(rules, transactions, min_confidence):
    satisfied_rules = []
    for rule in rules:
        conf = calculate_confidence(rule[0], rule[1], transactions)
        if conf >= min_confidence:
            satisfied_rules.append((rule[0], rule[1], conf))
    return satisfied_rules