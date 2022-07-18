import math
import statistics
from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from Utils.metautils import *

def oldcreatewindows(df, targetcol, windowsize, contextcol):
    windows = []
    df[targetcol] = df[targetcol].replace(0.00000, 0.01234)
    pricelist = list(df[targetcol])
    for backward, current in enumerate(range(len(pricelist)), start=0 - windowsize):
        if backward < 0:
            backward = 0
        window = pricelist[backward:current]
        l = len(window)
        if l == 0:
            window = [df[targetcol].mean()]
        window.extend([statistics.fmean(window)] * (96- l))
        window = [0 if math.isnan(x) else x for x in window]
        windows.append(window)

    df['window'] = windows
    simplerdf = df[['window', contextcol, targetcol]]
    df[targetcol] = df[targetcol].replace(0.01234, 0.00000)

    return simplerdf

def create_metafeaturedf(windowdf, features):
    print('calculating df meta-features')
    windowdf['MetaFeatureDict'] = windowdf.apply(lambda row: calcmetafeatures(row.window, features), axis=1)
    # windowdf['MetaFeatureVector'] = windowdf.apply(lambda row: (list(row.MetaFeatureDict.values())), axis=1)
    # windowdf['MetaFeatureVector'] = windowdf.apply(lambda row: [0 if math.isnan(x) else x for x in (row.MetaFeatureVector)], axis=1)
    # #windowdf['MetaFeatureVector'] = [MetaFeatureDict[x] for x in argv[4]]
    # metadf = windowdf.fillna(method="ffill")

    return windowdf

def normalise(trainmetadf, contextcol):
    df = pd.DataFrame(list(trainmetadf['MetaFeatureDict'])).set_index(trainmetadf.index)
    df.fillna(method="ffill")
    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df), columns = df.columns)
    # scaled[argv[3]] = list(trainmetadf[argv[3]])
    scaled[contextcol] = list(trainmetadf[contextcol])

    return scaled

def fit_model(cleanmetadf_train, contextcol, features, classification_alg):
    train_features = cleanmetadf_train[features] # .drop([targetcol], axis=1),
    # train_targets = list(cleanmetadf_train[argv[3]])
    train_targets = list(cleanmetadf_train[contextcol])
    model = classification_alg
    print('fitting')
    model.fit(train_features, train_targets)
    print('fitted')

    return model

def make_predictions(cleanmetadf_test, model, contextcol, features):
    test_features = cleanmetadf_test[features]
    print('predicting')
    predictions = model.predict_proba(test_features)
    print('predicted')
    predictionsdf = pd.DataFrame(predictions, columns=model.classes_)
    #predictionsdf['context'] = list(cleanmetadf_test[argv[3]])
    predictionsdf['context'] = list(cleanmetadf_test[contextcol])

    return predictionsdf