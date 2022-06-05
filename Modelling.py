import numpy as np
from matplotlib import pyplot as plt
from ModellingUtils import *
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge

period = '60T'
trainingfeatures = ['t-1', 'temp', 'Weekend', 'precip', 'uvindex'] #'U_Dt-1', 'D_Dt-1']
targetfeature = 'MID_PRICE'
model = svm.SVR()
# model = Ridge(random_state = 'randomstate', tol=1e-3, normalize=False, solver='auto')
#model = xgb.XGBRegressor(n_estimators=1000, eval_metric='mae', max_depth = 7,eta = .1, min_child_weight = 5, colsample_bytree = .4, reg_lambda = 50)
contextcol = 'season_cat'

infile = 'With_Features/'+ period + '/balance_2019_' + period
# infile = 'With_Features/Meteo/balance_2019Daily'
df = gatherdata(infile)
condict = createcontextdict(df, contextcol)
formodel = extractfeasandtargsfromcontextdict(condict, trainingfeatures, targetfeature)
final, reg = extractpredictions(formodel, model)
ratiodictionary = findcontextratio(contextcol, df)

avg_acc = 0
for context in final.keys():
    MSE = mean_squared_error(final[context][1], final[context][0]) #Y_true, y_pred
    print(context, MSE)
    xlength = len(final[context][0])
    plt.figure()
    plt.plot(np.arange(xlength), final[context][0], label='Predicted Price')
    plt.plot(np.arange(xlength), final[context][1], label='True Price')
    plt.legend()
    plt.title('Price Prediction for ' + str(context))
    plt.show()
    avg_acc += (MSE * ratiodictionary[context])

print(baseline(df, trainingfeatures, targetfeature, model))
print(avg_acc)