from Utils.NewMetaUtils import *

argv = [15, SVC(probability=True, verbose = True, max_iter = 50000), 'Target', 'consumption_context', ['mean']]#, 'std', 'kurtosis', 'adf', 'ac1', 'max', 'skew']] #remove argv[2]
features = argv[4] = ['mean', 'std', 'kurtosis', 'max']
#argv[2] = model = RandomForestClassifier(n_estimators=1000, max_depth=None, random_state=0)
fargv = ['WindowData/Minutely/' + argv[3] + '/Test', 'Windowdata/Minutely/' + argv[3] + '/Train']
# test = pd.read_pickle(fargv[0])

def splitdata(fargv, argv):
    df = pd.read_pickle(fargv[1]).sample(n=50000)
    df['window'] = list(df['window_' + str(argv[0])])
    df = df.drop(['window_15', 'window_30', 'window_60'], axis = 1)
    train = df.sample(frac=0.8, random_state=42)
    dev = df.drop(train.index)
    # test = pd.read_pickle(fargv[0]).sample(n =10000)

    return train, dev#, test

#Split
train, dev = splitdata(fargv, argv)

#MetaDFs
trainmetadf = create_metafeaturedf(train, argv[4])
devmetadf = create_metafeaturedf(dev,  argv[4])
# testmetadf = create_metafeaturedf(test,  argv[4])

#Normalise
cleanmeta_train = normalise(trainmetadf, argv[3])
cleanmeta_dev = normalise(devmetadf, argv[3])
# cleanmeta_test = normalise(testmetadf, argv[3])

#SaveMetaFeatureDF
outmeta_train = cleanmeta_train
outmeta_train['Target'] = list(trainmetadf['Target'])
os.makedirs('MetaFeatureData', exist_ok = True)
outmeta_train.to_pickle('MetaFeatureData/'+ argv[3])

#Fit
model = fit_model(cleanmeta_train, argv[3], argv[4], argv[1])

#Predict
devpredictionsdf = make_predictions(cleanmeta_dev, model, argv[3], argv[4])
testpredictionsdf = make_predictions(cleanmeta_test, model, argv[3], argv[4])

for i in range(len(devpredictionsdf.columns) - 1):
    print(i)
    for j in range(len(devpredictionsdf.columns) - 1):
        print(round(devpredictionsdf[devpredictionsdf['context'] == j].iloc[:, i].mean(), len(devpredictionsdf.columns)))

def make_pickle(predictionsdf):
    os.makedirs('Probpreds/' + argv[3] + '/' + str(argv[0]), exist_ok=True)
    pickleoutfile = 'Probpreds/' + argv[3] + '/' + str(argv[0]) + '/' + 'newest'
    predictionsdf.to_pickle(pickleoutfile)
    print('pickled created at', pickleoutfile)
    return

make_pickle(devpredictionsdf)



