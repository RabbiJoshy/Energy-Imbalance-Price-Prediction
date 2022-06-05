from metautils import *

argv = [96*7, 96, 2018, 'season_cat', ['mean', 'std', 'kurtosis', 'adf', 'ac1', 'max', 'skew']]
#argv = [96, 24, 2018, 'season_cat', ['mean', 'std', 'kurtosis', 'adf', 'ac1', 'max', 'skew']]

# argv[0], argv[1] = 96, 24
argv[4] = ['mean', 'std', 'kurtosis', 'skew']

fargv = ['With_Features/Meteo/balance_2019Daily', 'With_Features/Meteo/balance_2018Daily' ]
# windowsize = 96
# step = 24 #default24
# testyear = 2018
# trainingfeas = ['mean', 'std', 'kurtosis', 'adf', 'ac1']
# contextcol = 'season_cat'
# metadatadf = gettraininmetagdf(fargv[0], argv[0], argv[1], argv[3])
# TEST_metadatadf, TEST_contextlist, TEST_stamp_list = gettestmetadf(fargv[1], argv[0], argv[1], argv[3])
# finalweightdf = getweightdf(metadatadf, TEST_metadatadf, TEST_contextlist, TEST_stamp_list, argv[4])
#
# acronym = ''.join([word[0].upper() for word in argv[4]])
# pickleoutfile = 'Probpreds/' + str(argv[2]) + acronym + 'step' + str(argv[1]) + 'WS' + str(argv[0])
# finalweightdf.to_pickle(pickleoutfile)

def metafeatures(argv, fargv):
    print('getting trainging data')
    metadatadf = gettraininmetagdf(fargv[0], argv[0], argv[1], argv[3])
    TEST_metadatadf, TEST_contextlist, TEST_stamp_list = gettestmetadf(fargv[1], argv[0], argv[1], argv[3])
    finalweightdf = getweightdf(metadatadf, TEST_metadatadf, TEST_contextlist, TEST_stamp_list, argv[4])
    acronym = ''.join([word[0].upper() for word in argv[4]])
    # pickleoutfile = 'Probpreds/' + str(argv[2]) + acronym + 'step' + str(argv[1]) + 'WS' + str(argv[0])
    # pickleoutfile = 'Probpreds/' + acronym + '_St' + str(argv[1]) + '_WS' + str(argv[0])
    pickleoutfile = 'Probpreds/' + str(argv[1]) + '.' +str(argv[0]) + '/' + 'newest'
    finalweightdf.to_pickle(pickleoutfile)
    print('pickled created at', pickleoutfile)

    return finalweightdf

metafeatures(argv, fargv)
#
# for i in range(len(W) - window_size + 1):
#     print(W[i: i + window_size]

#do i make consistent wihndows all llikk 1 month or somecs specfic timelength (start with a day)
#so one option is to have a day with a sliding window, another option is to just have each day be a training point


