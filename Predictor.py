from PredictorUtils import *
import pandas as pd
from sklearn import svm

#TODO- deal with creating new folders for saving best models?, make sure predictions also take into account metafeaturecomb

fargv = ['With_Features/Meteo/balance_2019Daily',
         'With_Features/Meteo/balance_2018Daily',]

argv = [['t-1', 'temp', 'Weekend', 'precip', 'uvindex'],

        'MID_PRICE',
        'season_cat',
        2018,

        [svm.SVR(),
         LinearRegression(),
         Ridge(random_state='randomstate', tol=1e-3, normalize=False, solver='auto'),
         xgb.XGBRegressor(n_estimators=1000, eval_metric='mae', max_depth=7, eta=.1, min_child_weight=5,
                          colsample_bytree=.4, reg_lambda=50)
         ]

        ]
        #'U_Dt-1', 'D_Dt-1']

def Predictor(argv):

    placeholder = gatherdata(fargv[0])
    sc = trainfiletoscoredict(placeholder, argv[2], argv[0], argv[1], argv[4])
    predsdict = gettrainedmodelpercontext(fargv[0], fargv[1], sc, argv[0], argv[1])
    formattedpredsdict = predsdictformatting(predsdict, fargv[1])

    return formattedpredsdict


def createoutpickle(argv, formattedpredsdict):
    modelsacronym = ''.join([str(word)[0] for word in argv[4]])
    outfile = 'ContextPreds/' + modelsacronym + '/Newest'
    formattedpredsdict.to_pickle(outfile)
    print('preds pickle created at', outfile)

    return

formattedpredsdict = Predictor(argv)
createoutpickle(argv, formattedpredsdict)

# for i in range(10):
#     print(y_test[i])
#     print(y_pred[i])
