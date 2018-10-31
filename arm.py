import importcsv as ic
import processdata as procd
import pandas as pd
import fptree as fp


def preprocess_data_for_arm():
    
    data = ic.separateImport()
    data = procd.fillData(data, fill_method="median")

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
    return data

def generate_arm_rules():
    # accuracy of rule
    minsup = 0.4 # adjust
    transactions = preprocess_data_for_arm()
    for itemset in fp.find_frequent_itemsets(transactions, int(len(transactions)*minsup), include_support = True,df=True):
        if len(itemset[0])>1:
            print(itemset)
    return None

def experiments():
    

if __name__ == '__main__':
    # generate_arm_rules()
    find_out()

# to see if the results are the same as lecture example
def test():
    transactions = [['f','a','c','d','g','i','m','p'], ['f','a','b','c','l','m','o'], ['b','f','h','j','o'], ['b','c','k','s','p'], ['a','f','c','e','l','p','m','n']]
    for itemset in fp.find_frequent_itemsets(transactions, 3, include_support=True):
        print(itemset)