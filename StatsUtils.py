import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy import stats
import pandas as pd
import numpy as np

#PLOTS
def plotminmax(df, mins = 100):

    # x1 = inlier.index.tolist()[:mins]
    # y1 = inlier['MIN_PRICE'][:mins]
    # x2 = inlier.index.tolist()[:mins]
    # y2 = inlier['MAX_PRICE'][:mins]
    # plt.scatter(x1, y1, s=1, c = 'b')
    # plt.scatter(x2, y2, s = 1, c = 'r')

    df = df[:mins]
    ax = df.plot(x='Time', y='MID_PRICE', c='white')
    plt.fill_between(x='Time', y1='MIN_PRICE', y2='MAX_PRICE', data=df)

    plt.xticks(rotation=45)
    plt.legend(["min", "max"])
    plt.show()

def boxplots(x, y, data, significance = 'off'):
    box = data
    # plt.figure(figsize=(10, 6))
    # plt.text(75, 0.07, 'Outliers beyond beyond 75% value', fontsize=14)
    ax = sns.boxplot(x=x, y=y, data=box)
    if significance == 'on':
        sigs = t_test(data, x, y)
        print(sigs)
        df = pd.DataFrame(sigs, columns=['pair', 'significance'])
        df.plot(x='pair', y='significance', kind='bar')
        plt.xticks(rotation=25)
    else:
        sigs = 'SignificanceOff'
    # plt.show()

    return sigs


def t_test(df, col, pricecol):
    contexts = dict()
    for i in df[col].unique():
        print(i)
        contexts[i] = df[df[col]== i][pricecol].to_list()

    significant = []
    for comb in combinations(contexts.keys(), 2):
        print("Combination: ", comb)
        print(np.array(contexts[comb[0]])[:100])
        print(np.array(contexts[comb[1]])[:100])
        print(stats.ttest_ind(np.array(contexts[comb[0]])[:], np.array(contexts[comb[1]])[:], nan_policy='omit', equal_var=False)[1])

        if stats.ttest_ind(contexts[comb[0]], contexts[comb[1]], nan_policy='omit', equal_var=False)[1] < 0.05:
            significant.append((str(comb), stats.ttest_ind(contexts[comb[0]], contexts[comb[1]], nan_policy='omit', equal_var=False)[1]))

    print(significant)

    return significant



