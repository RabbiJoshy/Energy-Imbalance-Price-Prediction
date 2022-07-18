import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn import svm
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

def createcontextdict(df, contextcol = 'season_cat'):
    contextdict = dict()
    for context in list(df[contextcol].unique()):
        contextdict[context] = df[df[contextcol] == context]

    return contextdict

def trainfiletoscoredict(traindf, contextcol, trainingfeatures, targetfeature, models):
    condict = createcontextdict(traindf, contextcol)
    bycontextmodelsdict = dict()
    for context in condict.keys():
        print(context)
        X = condict[context][trainingfeatures]
        y = condict[context][targetfeature]
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.33, random_state=42)

        modelscoredict = dict()
        best_rMSE = 100000
        print(len(models))
        for model in models:
            print('fitting', str(model)[:5])
            regr = model
            model.fit(X_train, y_train)
            print('predicting', str(model)[:5])
            y_pred = model.predict(X_dev)
            rMSE = mean_squared_error(y_dev, y_pred)
            modelscoredict[str(model)] = rMSE
            print(rMSE)
            if rMSE < best_rMSE:
                best_rMSE = rMSE
                os.makedirs('models/' + str(contextcol) + '/' + str(context), exist_ok=True)
                filename = 'models/' + str(contextcol) + '/' + str(context) + '/best'
                pickle.dump(model, open(filename, 'wb'))
                print(str(model)[:5], 'model saved to', filename)

        print('finished models for ', context)
        bycontextmodelsdict[context] = modelscoredict

        df = pd.DataFrame.from_dict(bycontextmodelsdict)

    return df

def gettrainedmodelpercontext(trainfile, testfile, scoredf, trainingfeatures, targetfeature, contextcol):
    testdf = pd.read_pickle(testfile)    #.sample(frac=0.1, replace=False, random_state=1) #TODO remove this
    predictionsbycontextdict = dict()
    for context in scoredf.keys():  #season_cat.unique()
        traindf = pd.read_pickle(trainfile)
        traindf = traindf[traindf[contextcol] == context] #.sample(frac=0.1, replace=False, random_state=1) #TODO remove this
        X = traindf[trainingfeatures]
        y = traindf[targetfeature]
        X_test = testdf[trainingfeatures]
        print(y.isnull().sum())
        model = pickle.load(open('models/' + str(contextcol) + '/' + str(context) + '/best', 'rb'))
        print('fitting', context)
        regr = model.fit(X,y)
        print('predicting', context)
        predictions = regr.predict(X_test)
        predictionsbycontextdict[context] = predictions
        print('predicted')

    return predictionsbycontextdict, testdf

def predsdictformatting(predsdict, testdf, targetfeature = 'Target(1)'):
    predsdictdf = pd.DataFrame.from_dict(predsdict)
    #todo need to fix this cols thing so that it works for all conetxs
    colsraw = list(predsdict.keys())
    cols = [str(col) + '_Predictions' for col in colsraw]
    predsdictdf.set_axis(cols, axis=1,inplace=True)

    predsdictdf['True'] = list(testdf[targetfeature])
    predsdictdf['Time'] = testdf.index
    predsdictdf = predsdictdf.set_index('Time')

    return predsdictdf