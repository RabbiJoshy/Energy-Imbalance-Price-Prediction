import json
from Utils.PredictorUtils import *
from sklearn import svm
import os
#TODO - create a mapping from contexts to meta-features
#TODO bring back clusters
#TODO create a baseline
argv = [['MID_PRICE','Forecasted Load', 'PRICE_FORECAST', 'REGULATION_STATE'],
        'Target(1)',
        'meteo_context',
        2018, # this can be removed
        []]

fargv = ['Clusteringdata/' + argv[2] + '/Train', 'Clusteringdata/' + argv[2] + '/Test']

test = pd.read_pickle(fargv[0])

argv[4] = [svm.SVR(),
         LinearRegression(),
         Ridge(random_state='randomstate', tol=1e-3, solver='auto'),
         xgb.XGBRegressor(n_estimators=1000, eval_metric='mae', max_depth=7, eta=.1, min_child_weight=5, colsample_bytree=.4, reg_lambda=50)
         ]

predictors = json.load(open("Scripts/ClusterDict.pkl", "r"))[argv[2].replace('_context', "")][1]
argv[0] = ['MID_PRICE'] + predictors

def Predictor(argv):

    sc = trainfiletoscoredict(pd.read_pickle(fargv[0]).sample(n = 10000), argv[2], argv[0], argv[1], argv[4])
    predsdict, testdf = gettrainedmodelpercontext(fargv[0], fargv[1], sc, argv[0], argv[1], argv[2])
    formattedpredsdict = predsdictformatting(predsdict, testdf, 'Target(1)')

    return formattedpredsdict

def createoutpickle(argv, formattedpredsdict):
    modelsacronym = ''.join([str(word)[0] for word in argv[4]])
    os.makedirs('ContextPreds/' + argv[2] + '/' + modelsacronym, exist_ok=True)
    outfile = 'ContextPreds/' + argv[2] + '/' + modelsacronym + '/Newest'
    formattedpredsdict.to_pickle(outfile)
    print('preds pickle created at', outfile)

    return

formattedpredsdict = Predictor(argv)
createoutpickle(argv, formattedpredsdict)



# for i in range(10):
#     print(y_test[i])
#     print(y_pred[i])
