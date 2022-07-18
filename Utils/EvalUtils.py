import pandas as pd
import os
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

def Plot(df, cm, plot=None):
    # if plot == 'heatmap':
    #     sns.heatmap(cm, annot=True)
    #
    if plot == 'truevsweight':
        x = df.index
        y = df['True']
        z = df['Weighted_Predictions']
        plt.plot(x, y)
        plt.plot(x, z)
    #
    # if plot == 'seasonalmodels':
    #     cols = ["Winter_Predictions", "Summer_Predictions"]
    #     #cols.append('Spring_Predictions')
    #     #cols.append(Autumn_Predictions')
    #     # cols.append('True')
    #     df.plot(y= cols, use_index=True)
    #
    # plt.show()

def save_results(argv, df, cm):
    dfoutfile = 'Results/' + argv[4] + '/DF/' + argv[1] + '/' + argv[3]
    os.makedirs('Results/' + argv[4] + '/DF/' + argv[1], exist_ok=True)
    cmoutfile = 'Results/' + argv[4] + '/CM/' + argv[1] + '/' + argv[3]
    os.makedirs('Results/' + argv[4] + '/CM/' + argv[1], exist_ok=True)
    df.to_pickle(dfoutfile)
    # print('df saved to', dfoutfile)
    cmdf = pd.DataFrame(cm)
    cmdf.to_pickle(cmoutfile)
    # print('cm saved to', cmoutfile)

    return

def Weighted_Predictions_Error(probpredsdf, contextpredsdf):

    df = pd.concat([probpredsdf, contextpredsdf], axis = 1, join="inner")
    weightedpreds = []
    for i, row in df.iterrows():
        predsum = 0
        for context in df.context.unique():
            predsum += (row.loc[context] * row.loc[str(context) + '_Predictions'])
        weightedpreds.append(predsum)

    df['Weighted_Predictions'] = weightedpreds
    df['Weighted Error'] = df['Weighted_Predictions'] - df['True']

    return df

def highestpredictor(subdf):
    highestpred = []
    for i, row in subdf.iterrows():
        dropped = row.drop('context')
        probsonly = list(dropped)
        high = probsonly.index(max(probsonly))
        context = str(subdf.columns[high])
        highestpred.append(context)
    subdf['HighestPred'] = highestpred

    return subdf

def addclosest(subdf):
    closest = []
    for i, row in subdf.iterrows():
        dropped = row.drop('True')
        predsonly = list(dropped)
        true = row['True']
        differences = [abs(x - true) for x in predsonly]
        context = differences.index(min(differences))
        contextstring = str(subdf.columns[context]).replace('_Predictions','')
        closest.append(contextstring)
    subdf['Closest'] = closest

    return subdf

def confusmatrix_classreport(df, col1, col2):
    pred = df[col1]
    true = df[col2]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true, pred)
    from sklearn.metrics import classification_report
    cr = classification_report(true, pred, zero_division=0)

    return cm, cr