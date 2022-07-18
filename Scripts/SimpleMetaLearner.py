import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hdbscan
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
train = pd.read_pickle('MetaFeatureData/Train')
# preclustertest = pd.read_pickle('../Clustering/CleanData/Minutely/2019')[['Target']]
train = train.sample(n=20000)

kmeans = KMeans(n_clusters=5, random_state=0).fit(train.drop(['Target'], axis =1))
sns.scatterplot(data=train, x='mean', y = 'std',hue=kmeans.labels_)

X = train.iloc[:, :5]
y = list(train['Target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)

data_tuples = list(zip(y_test,y_pred, y_test - y_pred))
errors = pd.DataFrame(data_tuples, columns=['Month','Day', 'error'])