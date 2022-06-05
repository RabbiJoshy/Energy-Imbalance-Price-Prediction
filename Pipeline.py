from metafeatures import metafeatures
#TODO MAIN QUESTIONS: Am I going to use weather Data, am I gong to make the time series multivariate

#metaparams
argv = [96, 24, 2018, 'season_cat', ['mean', 'std', 'kurtosis', 'adf', 'ac1']]
fargv = ['With_Features/Meteo/balance_2019Daily', 'With_Features/Meteo/balance_2018Daily' ]

#modelparams
trainingfeatures = ['t-1', 'temp', 'Weekend', 'precip', 'uvindex']

weightdf = metafeatures(argv, fargv)
# Predictor()
# evaluation()