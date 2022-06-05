from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
#
# data = pd.read_pickle("With_Features/balance_2019")
# pricedata = data[['MID_PRICE', 'MIN_PRICE', 'Day_of_Week','Weekend']]
# pricedata['Time'] = data.index.tolist()


#Missing Values
def fillmv(df, meth = 'pad'):
    pricedata_filled = df.fillna(method = meth) # This needs to be discussed in data
    # print(pricedata_filled.describe())
    return pricedata_filled

# #Outlier Detection - Boxplot
# # sns.boxplot(x=pricedata_filled['MID_PRICE'])

# minutes = 1500 #1500 = 1day, 45000 = 1month, 450000, 527,040 in Leap Year

def Cluster(data, dependent = 'MID_PRICE', fraction = 1, k = 2):
    minutes = round(len(data) / fraction)
    mid_price_array = np.array(data[dependent])
    time_array = data.index.tolist()
    y = mid_price_array[:minutes]
    X = time_array[:minutes]

    kmeans = KMeans(n_clusters=k, random_state=0).fit(y.reshape(-1, 1))
    labs = kmeans.labels_

    return kmeans, labs, X, y, minutes

def remove_outlier(df, col = 'MID_PRICE', criterion = 2.5):
    z = np.abs(stats.zscore(df[col]))
    df['z_price'] = z
    inlier_prices = df[df['z_price'] < criterion]

    return inlier_prices

def remove_highlow(df, remove = 'none', threshold = 35):
    if remove == 'high':
        selection = df[df['MID_PRICE']< threshold]
    elif remove == 'low':
        selection = df[df['MID_PRICE'] >threshold]
    else:
        selection = df
    return selection

def selecthours(df, hours = ('00:00', '23:59')):
    if hours == 'day':
        time = ('08:40', '22:30')
    elif hours == 'night':
        time = ('22:30', '08:40')
    else:
        time = hours
    hourselection = df.between_time(time[0], time[1])

    return hourselection

def selectdays(df, datesbetween):

    mask = (df['Time'] > datesbetween[0]) & (df['Time'] <= datesbetween[1])
    dayselection = df.loc[mask]

    return dayselection

def plot(data,
         shape,
         dates = ('2019-1-2', '2019-1-10'),
         time = ('00:00', '23:59'),
         clusters=2,
         highlow = ('all', 35),
         weekend = 2,
         iv = 'SEQ_NR',
         dv = "MID_PRICE",
         ):

    print(dates)

    filled = fillmv(data)   #Fill

    selection = selecthours(filled, hours = (time[0], time[1])) #Hours

    selection = selectdays(selection, datesbetween = (dates[0], dates[1]))

    selection = remove_highlow(selection, remove = highlow[0], threshold= highlow[1]) #cut from threshold

    selection = remove_outlier(selection, 'MID_PRICE', 3)

    if weekend == 1:
        selection = selection[selection["Weekend"] == 1]
    elif weekend == 0:
        print("yes")
        selection = selection[selection["Weekend"] == 0]


    k, l, X, y, mins = Cluster(selection,fraction = 1, k=clusters) #Cluster then Plot
    # plt.scatter(X, y, c=l, s = 1)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x= iv,  # selection.index.tolist(),
                    y= dv,
                    s=10,
                    hue=l,
                    style= shape,
                    data=selection)
    plt.xlabel(iv)
    plt.ylabel(dv)


    plt.xticks(rotation=45)
    plt.show()
    print(len(y))

    return selection
#
# sel = plot(pricedata, ('2019-1-10', '2019-1-20'), highlow = ('all', 35), weekend = 0)

