from Clustering import *

mins = 100000
# data = pd.read_pickle("With_Features/Meteo/balance_2019")[:mins]
# data = pd.read_pickle('With_Features/Meteo/imbalance_2019Daily')[:mins]
#data = pd.read_pickle('With_Features/2D/imbalance_2019_2D')[:]
data = pd.read_pickle('With_Features/2D/balance_2019_2D')[:]

# balance = 'regstate'
# dependent = 'REGULATION_STATE'
balance = 'price'
dependent = 'MID_PRICE'



filled = fillmv(data, 'pad') #Fill before removing outlier
inlier = remove_outlier(filled, dependent, criterion = 3)
to_plot = filled #inlier


#
c = 'temp'#'Time'
y = dependent#'REGULATION_STATE' #'MID_PRICE'
x = 'dew' #'Weekend'


ax = to_plot.plot.scatter(x=x,
y=y,
c =c,
s = 1,
colormap='viridis',
rot = 45)
# ax2.legend(["A", "B"])
ax.set_xlabel(x)
ax.set_ylabel(y)
plt.show()


