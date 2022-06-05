import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import svm
from sklearn.metrics import mean_squared_error
import pandas as pd

def extractfeasandtargsfromcontextdict(condict, trainfeas, tarfea):
    split = dict()
    for context in condict.keys():
        features = condict[context][trainfeas]
        targets = condict[context][tarfea]
        X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                            test_size=0.33, random_state=42)
        split[context] = [X_train, X_test, y_train, y_test]


    return split

def extractpredictions(splitdict, regrmodel = 'SVM'):
    predtruthpercontext = dict()
    regrdict = dict()
    for context in splitdict.keys():
        regr = regrmodel
        regr.fit(splitdict[context][0], splitdict[context][2])
        y_pred = regr.predict(splitdict[context][1])
        predtruthpercontext[context] = [y_pred, splitdict[context][3]]
        regrdict[context] = regr

    return predtruthpercontext, regrdict

def traincontextmodels(contextdictionary, trainingfeatures, targetfeature, models):
    contextmodeldict = dict()
    for context in contextdictionary.keys():
        X = contextdictionary[context][trainingfeatures]
        y = contextdictionary[context][targetfeature]

        modeldict = dict()
        for model in models[:1]:
            print(str(model))
            regr = model
            fitted = regr.fit(X, y)
            print('fitting ', str(context), ' model')

            modeldict[str(model)] = fitted
        contextmodeldict[context] = modeldict

    return contextmodeldict


def baseline(rawdf, trainingfeatures, targetfeature, model):
    features = rawdf[trainingfeatures]
    targets = rawdf[targetfeature]
    X_train, X_test, y_train, y_test = train_test_split(features, targets,
                                                        test_size=0.33, random_state=42)
    regr = model
    regr.fit(X_train, y_train)
    y_pred = regr.predict(X_test)
    MSE = mean_squared_error(y_test, y_pred)

    return MSE

def findcontextratio(contextcol, wholedf):
    contexts = list(wholedf[contextcol].unique())
    ratiodict = dict()
    for context in contexts:
        ratio = (len(wholedf[wholedf[contextcol] == context]) )    /     len(wholedf)
        ratiodict[context] = ratio

    return ratiodict