import matplotlib.pyplot as plt

from EvalUtils import *
import seaborn as sns
#TODO - change the prediction window, add ENTSOE features, add meteo features?, add 2017, add meta-features, improve the context models
#argv = ['Probpreds/','24.96' , 'ContextPreds/', 'SLRX']
argv = ['Probpreds/','96.672' , 'ContextPreds/', 'SLRX']
#TODO I GUESS I COULD USE 2018 as the test becsause of global warming, so that summer doesnt perform better than winter year on year
def DoEvaluation(argv):

    plot = 'seasonalmodels'

    df = Evaluate(pd.read_pickle(argv[0]+argv[1] + '/Newest'), pd.read_pickle(argv[2]+ argv[3]+ '/Newest'))
    df = highestpredictor(df)
    df = addclosest(df)
    # cm2, cr2 = confusmatrix_classreport(df, 'Closest', 'context')
    cm, cr = confusmatrix_classreport(df, 'HighestPred', 'Closest')
    sorthighclose = df.sort_values(by = ['Closest', 'HighestPred'])

    dfoutfile = 'Results/DF/' + argv[1] +argv [3]
    cmoutfile = 'Results/CM/' + argv[1] +argv[3]
    df.to_pickle(dfoutfile)
    df.to_pickle(cmoutfile)
    print('matches are ', len(df[df['Closest'] == df['HighestPred']])/ len(df))

    if plot == 'heatmap':
        sns.heatmap(cm, annot=True)

    if plot == 'truevsweight':
        x = df.index
        y = df['True']
        z = df['Weighted_Predictions']
        plt.plot(x, y)
        plt.plot(x, z)

    if plot == 'seasonalmodels':
        cols = ["Winter_Predictions", "Summer_Predictions"]
        #cols.append('Spring_Predictions')
        #cols.append(Autumn_Predictions')
        # cols.append('True')
        df.plot(y= cols, use_index=True)

    plt.show()

    return df

df = DoEvaluation(argv)



def plotwin():
    A = np.sqrt(mean_squared_error(list(df['True']), list(df['Autumn_Predictions'])))
    Su = np.sqrt(mean_squared_error(list(df['True']), list(df['Summer_Predictions'])))
    Wi = np.sqrt(mean_squared_error(list(df['True']), list(df['Winter_Predictions'])))
    Sp = np.sqrt(mean_squared_error(list(df['True']), list(df['Spring_Predictions'])))
    W = np.sqrt(mean_squared_error(list(df['True']), list(df['Weighted_Predictions'])))

    objects = ('au', 'spri', 'suma', 'weighted', 'wint')
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, [A, Sp, Su, W, Wi], align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('MSE')
    plt.title('does it work?')
    plt.show

    return A, Su, Wi, Sp, W
# a, s, w, sp, wi = plotwin()






