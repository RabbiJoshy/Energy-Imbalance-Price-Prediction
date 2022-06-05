from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from Clustering import fillmv
import random

# data = pd.read_pickle("With_Features/Meteo/balance_2019")
#data = pd.read_pickle('With_Features/Meteo/imbalance_2019Daily')
data = pd.read_pickle('With_Features/14D/balance_2019_14D')[:]

mode = 'bal'

if mode == 'im':
    dependent = 'REGULATION_STATE'
elif mode == 'bal':
    dependent = 'MID_PRICE'

data['random'] = [random.randrange(1, 50, 1) for i in range(len(data))]
data['random2'] = [random.randrange(1, 50, 1) for i in range(len(data))]
pricedata = data
# pricedata = data[['temp', 'MID_PRICE', 'MIN_PRICE', 'Day_of_Week','Weekend', 'SEQ_NR', 'random', 'random2', 'season', 'solarradiation', vis]][:100000]
pricedata['Timenumber'] = [number ** 0.1 for number in np.arange(len(pricedata['Time']))]

clu = fillmv(pricedata)

cv1 = 'precip'
cv2 = dependent
cv3 = 'temp'
x = clu[[cv1, cv2, cv3]]

# kmeans = KMeans(n_clusters=4, random_state=0).fit(x)
# sns.scatterplot(data=x, x=cv1, y=cv2, hue=kmeans.labels_)
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = 'indigo', s = 150, alpha =0.4)
# plt.show()



# kmeans = KMeans(n_clusters=2, random_state=0).fit(x)
# data = clu
# data['cluster'] = kmeans.labels_
#
# data1 = data[data.cluster==0]
# data2 = data[data.cluster==1]
#
# kplot = plt.axes(projection='3d')
# # xline = np.linspace(0, 15, 1000)
# # yline = np.linspace(0, 15, 1000)
# # zline = np.linspace(0, 15, 1000)
# # kplot.plot3D(xline, yline, zline, 'black')
# # Data for three-dimensional scattered points
# kplot.scatter3D(data1.temp, data1.precip, data1.MID_PRICE, c='red', label = 'Cluster 1')
# kplot.scatter3D(data2.temp, data2.precip, data2.MID_PRICE,c ='green', label = 'Cluster 2')
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], color = 'indigo', s = 200)
# plt.legend()
# plt.title("Kmeans")
# plt.show()