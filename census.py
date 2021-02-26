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
    df = df.iloc[0:10]
    frequencies = frequency(df)
    
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
        freq_xy = frequencies[XY] 
        freq_x = frequencies[X]

        support_xy = freq_xy/total_records
        support_x = freq_x/total_records

        rule_confidence = support_xy/support_x


        
        print("Freq XY & X: ", freq_xy, freq_x)
        print("Support of XY & X: ", support_xy, support_x)
        print("Confidence: ", rule_confidence)

        confidence_rules_set[rule] = rule_confidence 
        # print(XY)

    print(confidence_rules_set)
    return sorted(confidence_rules_set, key=confidence_rules_set.get, reverse=True)


# def support(A, n, freqs):



def get_item_pairs(data, r = 2):
    """
    Returns a generator that yields item pairs, one at a time

    Parameters
    ----------
    data: iter(iter)
        A 2D iterable

    Returns
    -------
    Generator[(str, str), None, None] 
        A generator of 2 element tuples, format: Generator[YieldType, SendType, ReturnType]

    Raises
    ------
        None
    """
    # order_item = order_item.to_numpy()    

    #for dataframe
    for row in data.itertuples(index=False):
        for item_pair in combinations(row, r):
            yield item_pair        

def get_pairs(data, r = 2):
    for item_pair in combinations(data, r):
        # print(item_pair)
        yield item_pair        


def combinations_by_subset(seq, r):
    if r:
        for i in range(r - 1, len(seq)):
            for cl in combinations_by_subset(seq[:i], r - 1):
                yield cl + (seq[i],)
    else:
        yield tuple()

def get_combo(data):
    res = Counter()
    for row in data.itertuples(index=False):
        for i in range(2, len(row)+1):
            # new_row = Counter(combinations_by_subset(row, i))
            new_row = Counter(get_pairs(row, i))
            res += new_row
    return res

def get_combinations(data):
    attr_counts = Counter()
    for row in data.itertuples(index=False):
        for i in range(2, 4): #len(row)+1
            for item_pair in combinations(row, i):
                print("PAIR", item_pair)
                attr_counts[item_pair] += 1
    return attr_counts

def get_combinations2(data):
    for row in data.itertuples(index=False):
        for i in range(2, len(row)+1):
            for item_pair in combinations(row, i):
                yield item_pair   

def get_combinations3(data):
    for row in data.itertuples(index=False):
        n = len(row)
        return chain.from_iterable(combinations(row, r) for r in range(2, n+1))
 
# Returns frequency counts for items and item pairs
def frequency(iterable):
    # attribute_pairs = get_item_pairs(iterable)
    # attribute_pairs = get_combinations(iterable)
    # attribute_pairs = list(attribute_pairs)
    # attribute_pairs = Counter(chain.from_iterable(get_combinations2(iterable)))
    #print(type(attribute_pairs), type(attribute_pairs[0]))

    attribute_pairs = get_combo(iterable)
    return attribute_pairs
    # return pd.Series(attribute_pairs).rename("freq")

    # Counter iterates through our attribute pairs and keeps a tally of their occurrence
    attr_pair_count = Counter(attribute_pairs) # sort indicies? PerformanceWarning from get_combinations

    # print(attr_pair_count)

    return pd.Series(attr_pair_count).rename("freq")
    # df = pd.DataFrame.from_dict(d, orient='index').reset_index()

# Returns frequency and support associated with item
def merge_item_stats(item_pairs, item_stats):
    return (item_pairs
                .merge(item_stats.rename(columns={'freq': 'freqA', 'support': 'supportA'}), left_on='item_A', right_index=True)
                .merge(item_stats.rename(columns={'freq': 'freqB', 'support': 'supportB'}), left_on='item_B', right_index=True))

# Returns name associated with item
def merge_item_name(rules, item_name):
    columns = ['itemA','itemB','freqAB','supportAB','freqA','supportA','freqB','supportB', 
               'confidenceAtoB','confidenceBtoA','lift']
    rules = (rules
                .merge(item_name.rename(columns={'item_name': 'itemA'}), left_on='item_A', right_on='item_id')
                .merge(item_name.rename(columns={'item_name': 'itemB'}), left_on='item_B', right_on='item_id'))
    return rules[columns]     


def associationRules(data, min_support):
    print("Starting data: {:22d}".format(len(data)))

    # Calculate item frequency and support
    item_stats             = frequency(data).to_frame("freq")
    item_stats['support']  = item_stats['freq'] / len(data) 

    # print(item_stats)

    # Filter from data items below min support 
    qualifying_items       = item_stats[item_stats['support'] >= min_support].index
    df             = data[data.isin(qualifying_items)]

    # filtered = data['support'] >= min_support
    # print(item_stats)
    print(df)




if __name__ == '__main__':
    df = pd.read_csv('http://s3.amazonaws.com/istreet-questions-us-east-1/443606/census.csv', header = None)
    
    # df = df.to_numpy()[0:10]
    df = df.iloc[0:10]
    # print(df)

    # # Convert from DataFrame to a Series, with order_id as index and item_id as value
    # orders = orders.set_index('key')['product_id'].rename('item_id')

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

    # for i, row in x.iteritems():
    #     print(i, row) 

    # arrangingRules(rules)
    # print(associationRules(df, 0.3))

    # print(df.shape, df.shape[0])
    # for i in range(df.shape[0]):
    #     yield combinations(df[i], )

    # print(len(res))
    # print(res[check2])

    print(arrangingRules(rules))