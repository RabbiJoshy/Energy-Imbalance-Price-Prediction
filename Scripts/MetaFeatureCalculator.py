from Utils.NewMetaUtils import *

argv = [15, ['mean', 'std', 'kurtosis', 'max'], 'ignore']
train = pd.read_pickle('/Users/joshuathomas/Desktop/Thesis/Thesis Large Files/WindowData/Test')
train['window'] = list(train['window_' + str(argv[0])])
train[argv[2]] = [0]* len(train)

#MetaDFs
trainmetadf = create_metafeaturedf(train, argv[1])
cleanmeta_train = normalise(trainmetadf, argv[2])
cleanmeta_train['Target'] = list(train['Target'])
finaldf  = cleanmeta_train.drop([argv[2]], axis = 1)

finaldf.to_pickle('MetaFeatureData/Train')