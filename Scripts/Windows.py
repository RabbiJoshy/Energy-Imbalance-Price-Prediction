import os
import pandas as pd
import statistics
import math
preclustertrain = pd.read_pickle('../Clustering/CleanData/Minutely/2018')[['Target']]
preclustertest = pd.read_pickle('../Clustering/CleanData/Minutely/2019')[['Target']]
windowsizelist =[15,30,45]
windowcolnames = ['window_'+ str(win) for win in windowsizelist]

def createwindows(df, targetcol, windowsize):
    windows = []
    df[targetcol] = df[targetcol].replace(0.00000, 0.01234)
    pricelist = list(df[targetcol])
    for backward, current in enumerate(range(len(pricelist)), start=0 - windowsize):
        if current % 10000 == 0:
            print(current)
        if backward < 0:
            backward = 0
        window = pricelist[backward:current]
        l = len(window)
        if l == 0:
            window = [df[targetcol].mean()]
        window.extend([statistics.fmean(window)] * (windowsize- l))
        window = [0 if math.isnan(x) else x for x in window]
        windows.append(window)

    df[targetcol] = df[targetcol].replace(0.01234, 0.00000)

    return windows
trainwins = preclustertrain
for size in windowsizelist:
    windows = createwindows(preclustertrain, 'Target', size)
    trainwins['window_'+ str(size)] = windows

testwins = preclustertest
for size in windowsizelist:
    windows = createwindows(preclustertest, 'Target', size)
    testwins['window_'+ str(size)] = windows
#
os.makedirs('/Users/joshuathomas/Desktop/Thesis/Thesis Large Files/WindowData', exist_ok= True)
trainwins.to_pickle('/Users/joshuathomas/Desktop/Thesis/Thesis Large Files/WindowData/Train')
testwins.to_pickle('/Users/joshuathomas/Desktop/Thesis/Thesis Large Files/WindowData/Test')

def merge(windowdf, clusterdf):
    index_merged = windowdf[windowdf.index.isin(clusterdf.index)]
    cluster_added = pd.concat([index_merged, clusterdf], axis =1)

    return cluster_added

feasets = ['consumption_context', 'meteo_context', 'production_context']
trainwins = pd.read_pickle('/Users/joshuathomas/Desktop/Thesis/Thesis Large Files/WindowData/Train')

for feaset in feasets:
    clustertrain = pd.read_pickle('Clusteringdata/Minutely/' + feaset + '/Newest')
    window_cluster_train = merge(trainwins, clustertrain)
    window_cluster_train = window_cluster_train[windowcolnames + [feaset] + ['Target']]
    window_cluster_train = window_cluster_train.loc[:, ~window_cluster_train.columns.duplicated()].copy()
    os.makedirs('WindowData/Minutely/' + feaset, exist_ok=True)
    window_cluster_train.to_pickle('WindowData/Minutely/' + feaset + '/Train')
    # window_cluster_test.to_pickle('WindowData/Minutely/' + feaset + '/Test')




