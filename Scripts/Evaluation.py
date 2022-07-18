from Utils.EvalUtils import *
#TODO - change the prediction window, add ENTSOE features, add meteo features?, add 2017, add meta-features, improve the context models
#TODO 7/06 make it so that the contexts are unsupervsied, try out another context

#should i get the predictions to actually come from the closeness of tbe model in each case, does this even work


# TODO reallynimportant - makre sure the probspreds has the time series index

#argv = ['Probpreds/','24.96' , 'ContextPreds/', 'SLRX', 'season_cat', 'newest']
# argv = ['Probpreds/', '96.672','ContextPreds/', 'SLRX', 'season_cat', 'onehot']
argv = ['Probpreds/', '96','ContextPreds/', 'SLRX', 'meteo_context', 'newest']

def DoEvaluation(argv, plot = None):
    probsdf = pd.read_pickle(argv[0] + argv[4] + '/' + argv[1] + '/' + argv[5])
    predsdf = pd.read_pickle(argv[2] + argv[4] + '/' + argv[3] + '/Newest')
    probsdf.index = predsdf.index
    print('creating preds df')
    predsdf = addclosest(predsdf)
    print('creating probs df')
    probsdf = highestpredictor(probsdf)
    print('weighting')
    df = Weighted_Predictions_Error(predsdf, probsdf)
    df['context'] = df['context'].apply(str)

    # cm, cr = confusmatrix_classreport(df, 'HighestPred', 'Closest')
    # cm2, cr2 = confusmatrix_classreport(df, 'Closest', 'context')
    # cm3, cr3 = confusmatrix_classreport(df, 'HighestPred', 'context')
    #
    # save_results(argv, df, cm)
    # Plot(df, cm)
    #
    print(argv[1], argv[5])
    print('closest/hipredicted (accuracy of using optimal model) matches: ',
          len(df[df['Closest'] == df['HighestPred']]) / len(
              df))  # this is actually what accuracy is asnd weighted recall
    print('closest/context (Is the actual context actually the best?): ',
          len(df[df['Closest'] == df['context']]) / len(df))
    print('context/hipredicted (does it predict using the actual context?): ',
          len(df[df['context'] == df['HighestPred']]) / len(df))
    print('rMSE = ', np.sqrt(mean_squared_error(df['True'], df['Weighted_Predictions'])))
    print("")

    return df #, cm, cr, cr3, cr2

# for windowstep in ['24.96']: #, '96.672']:
#     argv[1] = windowstep
#     for encoding in ['onehot']: #, 'newest']:
#         argv[5] = encoding
#         DoEvaluation(argv)

df = DoEvaluation(argv)
# df = df.sort_values(by = ['context'])





# def plotwin():
#     A = np.sqrt(mean_squared_error(list(df['True']), list(df['Autumn_Predictions'])))
#     Su = np.sqrt(mean_squared_error(list(df['True']), list(df['Summer_Predictions'])))
#     Wi = np.sqrt(mean_squared_error(list(df['True']), list(df['Winter_Predictions'])))
#     Sp = np.sqrt(mean_squared_error(list(df['True']), list(df['Spring_Predictions'])))
#     W = np.sqrt(mean_squared_error(list(df['True']), list(df['Weighted_Predictions'])))
#
#     objects = ('au', 'spri', 'suma', 'weighted', 'wint')
#     y_pos = np.arange(len(objects))
#     plt.bar(y_pos, [A, Sp, Su, W, Wi], align='center', alpha=0.5)
#     plt.xticks(y_pos, objects)
#     plt.ylabel('MSE')
#     plt.title('does it work?')
#     plt.show
#
#     return A, Su, Wi, Sp, W
# a, s, w, sp, wi = plotwin()






