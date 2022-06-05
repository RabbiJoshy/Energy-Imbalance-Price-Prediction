import numpy as np
import pandas as pd
from scipy.stats import kurtosis
import more_itertools
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from statsmodels.tsa.stattools import adfuller
# from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from scipy.stats import skew

def createcontextdictwindow(df, contextcol='season_cat', window=96):
    contextdictwindow = dict()
    for context in list(df[contextcol].unique()):
        contextdictwindow[context] = list(df[df[contextcol] == context]['MID_PRICE'])

    return contextdictwindow

def calcmetafeatures(window, context, usedstamps):
    metafeatures = dict()
    filteredwindow = list(filter(None, window)) #TODO I Idont want to  have to filter
    # metafeatures['context'] = context #this puts the context as a meta-feature
    metafeatures['mean'] = np.mean(filteredwindow)
    metafeatures['std'] = np.std(filteredwindow)
    metafeatures['kurtosis'] = kurtosis(filteredwindow)
    metafeatures['adf'] = adfuller(filteredwindow)[0]
    metafeatures['skew'] = skew(filteredwindow)

    acf = sm.tsa.acf(filteredwindow, nlags=3)
    metafeatures['ac1'] = acf[1]
    metafeatures['ac2'] = acf[2]
    metafeatures['max'] = max(filteredwindow)
    metafeatures['min'] = min(filteredwindow)



    # Additive Decomposition #TODO gives me statsmodels objects
    # data_tuples = list(zip(usedstamps, filteredwindow))
    # wts = pd.DataFrame(data_tuples, columns=['Time', 'point']).set_index('Time')
    #metafeatures['decomposed'] = seasonal_decompose(wts, model='additive', extrapolate_trend='freq')

    return metafeatures

def contextlist_to_mflist(windowlist, context, usedstamps):
    mflist = []
    for step in windowlist:
        mflist.append(calcmetafeatures(step, context, usedstamps))
    return mflist

# def contextdict_to_mflistdict(contextdictionary, windowsize = 96, step = 24):
#     mflistdict = dict()
#     for context in contextdictionary.keys():
#         testlist = contextdictionary[context][1]
#         # print(testlist)
#         testwindows = list(more_itertools.windowed(testlist, n=windowsize, step=step))
#         mflistdict[context]= contextlist_to_mflist(testwindows, context)
#
#     return mflistdict


def contextdict_to_mflistdict2(contextdictionary, windowsize=96, step=24):
    mflistdict = dict()
    stampsdict = dict()
    # stamplist = []
    for context in contextdictionary.keys():
        testlist = contextdictionary[context][1]
        testwindows = list(more_itertools.windowed(testlist, n=windowsize, step=step))
        testwindows = [[0 if j!=j else j for j in i] for i in testwindows] #TODO There is a nan sometimes in the testwindows


        stamps = contextdictionary[context][0]
        # trial = stamps[::24][:int(((8736-96)/24)+1)]
        matchedindices = []
        matchedwindows = list(more_itertools.windowed(range(len(testlist)), n=windowsize, step=step))
        for window in matchedwindows:
            matchedindices.append(window[0])
        usedstamps = [stamps[index] for index in matchedindices]
        stampsdict[context] = usedstamps
        # stamplist.append(usedstamps)

        mflistdict[context] = contextlist_to_mflist(testwindows, context, usedstamps)

    return mflistdict, stampsdict

def stampdict_to_stamplist(stampdictionary):
    stamplist = []
    contextlist = []
    for context in stampdictionary.keys():
        for stamp in stampdictionary[context]:
            stamplist.append(stamp)
            contextlist.append(context)
    return stamplist, contextlist

def createmetatrainingdata(mfvectordict):

    trainingdata = []
    for context in mfvectordict.keys():
        vecs = mfvectordict[context]
        for mfdict in vecs[:]:#TODO AUTOMATE THE FEATURES HERE
             trainingdata.append([mfdict['mean'],
                                  mfdict['std'],
                                  mfdict['kurtosis'],
                                  mfdict['adf'],
                                  mfdict['ac1'],
                                  mfdict['ac2'],
                                  mfdict['skew'],
                                  mfdict['max'],
                                  mfdict['min'],
                                  context])

    df = pd.DataFrame(trainingdata,columns=['mean', 'std', 'kurtosis',
                                            'adf', 'ac1', 'ac2', 'skew',
                                            'max', 'min',
                                            'context']) #, 'decomposed'])
    # metadf = df.dropna(axis=0) #TODO find out why the fuck there are NANs in the df
    metadf = df.fillna(df.mean()) #TODO findout how to get an average of the spring only not just the whole column

    return metadf


def metamodel(metatrainingdata, metatestingdata, trainingfeatures, targetcol='context'):  # TODO obviously I can improve the meta-model, but what is the purpose? do I want it ot be good?
    # X_train, X_test, y_train, y_test = train_test_split(metatrainingdata.drop([targetcol], axis=1),
    #                                                     metatrainingdata[targetcol], test_size=0.0,
    #                                                     random_state=101)
    # model = SVC(probability=True)
    # model.fit(X_train, y_train)
    # predictions = model.predict_proba(X_test)
    # print(classification_report(y_test, predictions))

    train_features = metatrainingdata[trainingfeatures]  # .drop([targetcol], axis=1),
    train_targets = metatrainingdata[targetcol]
    model = SVC(probability=True)
    model.fit(train_features, train_targets)
    print('fitting')
    # predictions = model.predict_proba(train_features)
    # predictionsdf = pd.DataFrame(predictions, columns=model.classes_)

    test_features = metatestingdata[trainingfeatures]
    predictions = model.predict_proba(test_features)
    print('predicting')
    # # print(classification_report(y_test, predictions))
    predictionsdf = pd.DataFrame(predictions, columns=model.classes_)

    return predictionsdf

def timestamp(df, contextcol='season_cat'):
    contextdict = dict()
    for context in list(df[contextcol].unique()):
        contextdict[context] = (list(df[df[contextcol] == context].index), list(df[df[contextcol] == context]['MID_PRICE']))

    return contextdict

def gettraininmetagdf(trainfile, windowsize, step, contextcol):
    df = pd.read_pickle(trainfile)
    condict = timestamp(df, contextcol)
    testmfsvectdict, stampsdict = contextdict_to_mflistdict2(condict, windowsize = windowsize, step = step)
    stamp_list, contextlist = stampdict_to_stamplist(stampsdict)
    metadatadf = createmetatrainingdata(testmfsvectdict)

    return metadatadf

def gettestmetadf(testfile, windowsize, step, contextcol):
    TEST_df = pd.read_pickle(testfile)
    TEST_condict = timestamp(TEST_df, contextcol)
    TEST_testmfsvectdict, TEST_stampsdict = contextdict_to_mflistdict2(TEST_condict, windowsize = windowsize, step = step)
    TEST_stamp_list, TEST_contextlist = stampdict_to_stamplist(TEST_stampsdict)
    TEST_metadatadf = createmetatrainingdata(TEST_testmfsvectdict)

    return TEST_metadatadf, TEST_contextlist, TEST_stamp_list

def getweightdf(metadatadf, TEST_metadatadf, TEST_contextlist, TEST_stamp_list, trainingfeas):
    weightingdf = metamodel(metadatadf, TEST_metadatadf, trainingfeas)
    weightingdf['context'] = TEST_contextlist
    weightingdf['Timestamp'] = TEST_stamp_list
    final = weightingdf.set_index('Timestamp')

    return final