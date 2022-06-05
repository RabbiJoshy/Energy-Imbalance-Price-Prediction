import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def Evaluate(probpredsdf, contextpredsdf):
    df = pd.concat([contextpredsdf, probpredsdf], axis = 1, join="inner")
    weightedpreds = []
    contexts = ['Autumn', 'Summer', 'Winter', 'Spring']
    for i, row in df.iterrows():
        predsum = 0
        for context in contexts:
            predsum += (row[context] * row[context + '_Predictions'])
        weightedpreds.append(predsum)

    df['Weighted_Predictions'] = weightedpreds
    df['Error'] = df['Weighted_Predictions'] - df['True']

    return df
def highestpredictor(df):
    highestpred = []
    for i, row in df.iterrows():
        mylist = list(row.iloc[[5,6,7,8]])
        hi = mylist.index(max(mylist))
        sea = df.columns[hi+5][:6]
        highestpred.append(sea)
    df['HighestPred'] = highestpred

    return df

def addclosest(df):

    closest = []
    for i, row in df.iterrows():
        a = list(row.iloc[[0,1,2,3]])
        true = row['True']
        a = [x - true for x in a]
        sea = a.index(min(a))
        closest.append(df.columns[sea][:6])

    df['Closest'] = closest

    return df

def confusmatrix_classreport(df, col1, col2):
    pred = df[col1]
    true = df[col2]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true, pred)
    from sklearn.metrics import classification_report
    cr = classification_report(true, pred)

    return cm, cr