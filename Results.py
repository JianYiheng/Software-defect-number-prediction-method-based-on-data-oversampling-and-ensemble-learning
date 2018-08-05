import re
import pandas as pd

def deposite(file):
    f = open(file, "r").read()
    SB = re.compile(r'(?<=Smote & Bagging ).+')
    S = re.compile(r'(?<=Smote )\w.+', re.M)
    B = re.compile(r'^Bagging.+', re.M)
    N = re.compile(r'(?<=Normal ).+')
    SB_f = SB.findall(f)
    S_f = S.findall(f)
    B_f = B.findall(f)
    N_f = N.findall(f)

    temp = []
    for i in B_f:
        temp.append(i[8:])
    B_f = temp

    results = pd.DataFrame([SB_f, S_f, B_f, N_f]).T
    results.columns = ['SMOTE&Bagging', 'SMOTE', 'Bagging', 'Normal']
    return results


if __name__ == "__main__":
    for i in ['Bayes.txt', 'DescionTree.txt', 'Linear.txt']:
        deposite(i).to_csv(i+'.csv')
