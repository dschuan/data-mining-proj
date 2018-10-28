import importcsv as ic
import processdata as procd
import pandas as pd
import fptree as fp

"""
Outline:
    
    -Process data such that all continuous variables are binned into categorical variables and are distinct items based on the attribute. 1 for attribute 1 should be att1_1 and 1 for attribute 2 should be att2_1

    -Collect all patterns that end at heart disease and all that do not

"""
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

    data_yes = data.loc[data['prediction'] >= 1]
    data_no = data.loc[data['prediction'] == 0]
    # return data_yes, data_no
    return data

def generate_arm_rules():
    # data_yes, data_no = preprocess_data_for_arm()
    data = preprocess_data_for_arm()
    for itemset in fp.find_frequent_itemsets(data, 5):
        print(itemset)
    # print("no presence of heart disease")
    # for itemset in fp.find_frequent_itemsets(data_no, len(data_no)/4):
    #     print(itemset)

    # print("presence of heart disease")
    # for itemset in fp.find_frequent_itemsets(data_yes, len(data_yes)/4):
    #     print(itemset)

if __name__ == '__main__':
    generate_arm_rules()
    