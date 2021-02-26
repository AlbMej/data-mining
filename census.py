import pandas as pd
import re
from itertools import chain, combinations, permutations
from collections import Counter


# The function is expected to return a STRING_ARRAY.
# The function accepts STRING_ARRAY rules as parameter.

def arrangingRules(rules):
    """
    Rearranges the given set of rules X=>Y, whe X and Y describe the attributes set.

    Args:
        rules: a list[str] of rules

    Returns:
        list[str]

    Raises:
        None
    """
    df = pd.read_csv('http://s3.amazonaws.com/istreet-questions-us-east-1/443606/census.csv', header = None)
    # df = df.iloc[0:10]

    _, dataset_supports = apriori(df.to_numpy(), 0.3)

    total_records = len(df)
    confidence_rules_set = {}

    for rule in rules:
        # Extracts both rules X & Y from the form {X}=>{Y}.
        X_Y = re.search(r"{(.*?)}=>{(.*?)}", rule).groups()
        
        X = tuple(X_Y[0].split(",")) # set of attributes
        Y = X_Y[1] # attributes, set?
        # Support(X): total number of records containing X / total number of records
        # Confidence(X=>Y): number of records containing X&Y / total number of records containing Y
        # Confidence(X=>Y): Support(X U Y) / Support(X)

        XY = X + (Y, )
        support_xy = dataset_supports[frozenset(XY)] 
        support_x = dataset_supports[frozenset(X)] 

        rule_confidence = support_xy/support_x
        confidence_rules_set[rule] = rule_confidence 

        # print("Support of XY & X: ", support_xy, support_x)
        print("Confidence: ", rule_confidence)

        confidence_rules_set[rule] = rule_confidence 
        # print(XY)

    # print(confidence_rules_set)
    return sorted(confidence_rules_set, key=confidence_rules_set.get, reverse=True)

 
def find_individual_attributes(dataset):
    # finds attribute sets
    attributes = set()
    for row in dataset:
        for attr_val in row:
            if (attr_val,) not in attributes:
                attributes.add((attr_val,))
    list(attributes)
    result = [frozenset(i) for i in attributes]
    return result


def filter_set(dataset, attribute_sets, min_support):
    frequencies = Counter()
    dataset = [set(row) for row in dataset] # Speed up subset check
    total_records = len(dataset)
    for row in dataset:
        for cur_set in attribute_sets:
            # If our current set is a subset in our row 
            if cur_set.issubset(row):
                # Record frequency of attribute=value set
                frequencies[cur_set] += 1
    attribute_supports = {}
    reduced_attributes = [] # List of our attributes post pruning

    for attr_set in frequencies:
        support_of_attr = frequencies[attr_set]/total_records
        # Check support meets minimum requirements, 
        if support_of_attr >= min_support:
            # Record support in dictionary and corresponding set in list
            attribute_supports[attr_set] = support_of_attr
            reduced_attributes.append(attr_set)

    return attribute_supports, reduced_attributes


def create_combinations(attribute_list, val_idx):
    new_sets = []
    # Compare every attribute with every other attribute
    for i in range(len(attribute_list)):
        for j in range(i+1, len(attribute_list)):
            cur_attr, other_attr = list(attribute_list[i]), list(attribute_list[j])
            # Sort to get head of attributes
            cur_attr.sort()
            other_attr.sort()
            # Use val_idx to find head of attributes
            cur_attr, other_attr = cur_attr[:val_idx-2], other_attr[:val_idx-2]

            if cur_attr ==  other_attr:
                # Keep recurring attribute sets
                combined_set = attribute_list[i].union(attribute_list[j])
                new_sets.append(combined_set)
    return new_sets

def apriori(dataset, min_support=0.3):
    size_1_sets = find_individual_attributes(dataset)
    # 1st_pruned, support
    attr_supports, pruned_attrs = filter_set(dataset, size_1_sets, min_support) 
    attr_list = [pruned_attrs]
    valid_attrs_index = 2

    # Continue increasing set sizes while pruning away attributes below min support
    notempty = len(attr_list[valid_attrs_index-2])
    while (notempty): # Continue until we reach an empty attribute set
        potential_attr_sets = create_combinations(attr_list[valid_attrs_index-2], valid_attrs_index)
        new_supports, new_pruned = filter_set(dataset, potential_attr_sets, min_support) 
        attr_supports.update(new_supports)
        attr_list.append(new_pruned)
        valid_attrs_index += 1
        notempty = len(attr_list[valid_attrs_index-2])
    return attr_list, attr_supports

if __name__ == '__main__':
    check1 = ('capital-gain=None', 'capital-loss=None')
    check2 = ('native-country=United-States', 'capital-gain=None', 'capital-loss=None')
    rule1 = "{native-country=United-States,capital-gain=None}=>{capital-loss=None}"
    rule2 = "{capital-gain=None,capital-loss=None}=>{native-country=United-States}"
    rule3 = "{native-country=United-States,capital-loss=None}=>{capital-gain=None}"
    rules = [rule1, rule2, rule3]

    # x = frequency(df)
    # # print(type(x))
    # print(x[check1])
    # # print(x)

    # print(len(x))
    # x.to_csv("result")

    # for i, row in df.iteritems():
    #     print(i, row.index) 

    # print(list(df.columns.values))
    # arrangingRules(rules)
    # print(associationRules(df, 0.3))

    # print(df.shape, df.shape[0])
    # for i in range(df.shape[0]):
    #     yield combinations(df[i], )

    # print(len(res))
    # print(res[check2])

    print(arrangingRules(rules))