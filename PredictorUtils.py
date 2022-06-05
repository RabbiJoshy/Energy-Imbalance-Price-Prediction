import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
import pickle
import pandas as pd
from sklearn import svm
from Clustering import fillmv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def createcontextdict(df, contextcol = 'season_cat'):
    contextdict = dict()
    for context in list(df[contextcol].unique()):
        contextdict[context] = df[df[contextcol] == context]

    return contextdict

def gatherdata(featurepickle):
    df = pd.read_pickle(featurepickle)
# df = pd.read_pickle('With_Features/Meteo/balance_2019Daily')[:]
    df['t-1'] = df['MID_PRICE'].shift(+1, fill_value=df["MID_PRICE"].mean())
    df['t-2'] = df['MID_PRICE'].shift(+2, fill_value=df["MID_PRICE"].mean())

    df['U_Dt-1'] = df['UPWARD_DISPATCH'].shift(+1, fill_value=df['UPWARD_DISPATCH'].mean())
    df['D_Dt-1'] = df['DOWNWARD_DISPATCH'].shift(+1, fill_value=df['DOWNWARD_DISPATCH'].mean())

    df = fillmv(df)

    return df

def trainfiletoscoredict(traindf, contextcol, trainingfeatures, targetfeature, models):
    condict = createcontextdict(traindf, contextcol)
    bycontextmodelsdict = dict()
    for context in condict.keys():
        print(context)
        X = condict[context][trainingfeatures]
        y = condict[context][targetfeature]
        X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.33, random_state=42)

        modelscoredict = dict()
        best_rMSE = 0
        for model in models:
            print('fitting', str(model))
            regr = model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_dev)
            rMSE = mean_squared_error(y_dev, y_pred)
            modelscoredict[str(model)] = rMSE
            if rMSE > best_rMSE:
                best_rMSE = rMSE
                filename = 'models/' + context + '/best'
                pickle.dump(model, open(filename, 'wb'))
                print(str(model), 'model saved to', filename)

        print('finished models for ', context)
        bycontextmodelsdict[context] = modelscoredict

        df = pd.DataFrame.from_dict(bycontextmodelsdict)

    return df

def gettrainedmodelpercontext(trainfile, testfile, scoredf, trainingfeatures, targetfeature):
    testdf = gatherdata(testfile)
    predictionsbycontextdict = dict()
    for context in scoredf.keys():
        traindf = gatherdata(trainfile)
        traindf = traindf[traindf['season_cat'] == context]
        X = traindf[trainingfeatures]
        y = traindf[targetfeature]
        X_test = testdf[trainingfeatures]
        y_test = testdf[targetfeature].fillna(method="ffill") #TODO explain NANs
        print(y.isnull().sum())
        model = pickle.load(open('models/' + context + '/best', 'rb'))
        regr = model.fit(X,y)
        print('predicting', context)
        predictions = regr.predict(X_test)
        predictionsbycontextdict[context] = predictions

    return predictionsbycontextdict

def predsdictformatting(predsdict, testfile):
    predsdictdf = pd.DataFrame.from_dict(predsdict)
    cols = ['Winter_Predictions', 'Spring_Predictions', 'Summer_Predictions', 'Autumn_Predictions']
    predsdictdf.set_axis(cols, axis=1,inplace=True)
    predsdictdf['True'] = list(gatherdata(testfile)['MID_PRICE'])
    predsdictdf['Time'] = gatherdata(testfile).index
    predsdictdf = predsdictdf.set_index('Time')

    return predsdictdf