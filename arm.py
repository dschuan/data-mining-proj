import importcsv as ic
import processdata as procd
import pandas as pd
import fptree as fp
from itertools import chain, combinations

# we want to drop the gender in our data set because it is disproportionately represented. In ARM, this is significant because ARM is determined by the frequency of itemset.
gender_bias = True # mitigated if we use fill_method as none but this reduces dataset size
minconf = 0.4
minsup = 0.5

#puts data into bins and rename the values so that it is readable
def preprocess_data_for_arm():
    
    data = ic.separateImport()
    data = procd.fillData(data, fill_method='none', exclude_col=False)

    num_age_groups = 3 #min is 29 and max is 77
    data["age"] = pd.cut(data["age"], num_age_groups, labels = [str(i) for i in range(num_age_groups)])
    
    num_trestbps_groups = 3 #min is 94 and max is 200
    data["trestbps"] = pd.cut(data["trestbps"], num_age_groups, labels = [str(i) for i in range(num_trestbps_groups)])

    num_chol_groups = 3 #min is 126 and max is 564
    data["chol"] = pd.cut(data["chol"], num_chol_groups, labels = [str(i) for i in range(num_chol_groups)])

    num_oldpeak_groups = 3 #min is 0 and max is 6.2
    data["oldpeak"] = pd.cut(data["oldpeak"], num_oldpeak_groups, labels = [str(i) for i in range(num_oldpeak_groups)])

    num_thalach_groups = 3 #min is 71 and max is 202
    data["thalach"] = pd.cut(data["thalach"], num_thalach_groups, labels = [str(i) for i in range(num_thalach_groups)])
    
    # attach string to all values except prediction column
    for label in ic.LABELS[:-1]:
        data[label] = label + " " + data[label].astype(str)

    data['prediction'] = ["no heart disease" if x == 0 else "heart disease" for x in data['prediction']]
    if gender_bias:
        data = data.drop(columns = ['sex'])
    return data

# generates frequent itemsets using fptree
def generate_frequent_itemsets():
    transactions = preprocess_data_for_arm()
    for itemset in fp.find_frequent_itemsets(transactions, int(len(transactions)*minsup), include_support = True,df=True):
        yield itemset

#generates rules with at least minimum confidence based on frequent itemsets generated
def generate_confidence():
    results = [i for i in generate_frequent_itemsets()]
    # we want to iterate through the largest itemsets first to set up pruning
    sorted_results = sorted(results, key=lambda a:len(a[0]), reverse=True)
    # we also want to keep the counts of all the transactions in a dict for easy access. Here the list is sloppily converted into a string
    results_dict = dict()
    for itemset, count in sorted_results:
        results_dict[''.join(i for i in itemset)] = count
    for itemset, count in sorted_results:
        # since items in the itemset are already L-ordered (or by frequency) then we should always check the first item first which is confidence ( first item | given the rest) because if the confidence is below minconf, we can just prune the checking of the other items since their frequency is lower
        if len(itemset) < 2:
            # no point going through itemsets with only 1 item
            break
        for item in powerset(itemset):
            # power set returns tuples so we convert tuples to list
            # item will always go from most frequent to least frequent
            result = list(item)
            given = itemset.copy()
            try:
                for i in result:
                    given.remove(i)
                if (given is None or len(given) < 1):
                    break
                key_itemset = ''.join(i for i in given)
                _count = results_dict[key_itemset]
                confidence = count/_count
                if confidence > minconf:
                    yield (given, result, confidence)
                else:
                    ##By anti-monotone property, we can stop the iteration here and prune the rest of the combinations
                    break

            except KeyError:
                # if result is not found in dict, it's not frequent enough and will not even be considered
                break

#prints rules
def print_rules():
    rules = list(generate_confidence())
    rules.sort(key=lambda rule: rule[2], reverse=True)
    for given, rule, confidence in rules:
        print("{} --> {} confidence: {:.2%}".format(given, rule, confidence))

#prints frequent itemset
def print_frequent_itemsets():
    for itemset in generate_frequent_itemsets():
        print(itemset)


def test_sex():
    data = preprocess_data_for_arm()
    print(len(data))
    print(len(data.loc[data['sex']=='sex 1.0']))

def powerset(iterable):
    # powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    s = list(iterable)
    # convert to list and omit empty tuple
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))[1:]

# to see if the results are the same as lecture example
def test():
    transactions = [['f','a','c','d','g','i','m','p'], ['f','a','b','c','l','m','o'], ['b','f','h','j','o'], ['b','c','k','s','p'], ['a','f','c','e','l','p','m','n']]
    for itemset in fp.find_frequent_itemsets(transactions, 3, include_support=True):
        print(itemset)

if __name__ == '__main__':
    print_rules()
    # print_frequent_itemsets()
    # test_sex()