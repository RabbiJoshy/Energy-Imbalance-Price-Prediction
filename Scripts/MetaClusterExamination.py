import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import hdbscan
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
feaset = 'consumption_context'

df = pd.read_pickle('MetaFeatureData/' + feaset)

X = df.iloc[:, :5]
y = list(df['Target'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,y_pred)

data_tuples = list(zip(y_test,y_pred, y_test - y_pred))
errors = pd.DataFrame(data_tuples, columns=['Month','Day', 'error'])







#MetaCluster?
df.fillna(method="ffill")
unreduced = df.sample(n = 1000)
# sns.scatterplot(data=unreduced, x='std', y = 'max',hue=feaset)
# sns.scatterplot(data=unreduced, x='max', y = 'mean',hue=feaset)
# sns.scatterplot(data=unreduced, x='mean', y = 'kurtosis',hue=feaset)
unreduced = df.sample(n = 6000)
labels = list(unreduced[feaset])
unreduced_feasonly = unreduced.drop([feaset], axis =1)
#Reduction

T = TSNE(n_components=2, verbose = 1)
reduced = T.fit_transform(unreduced_feasonly)


pca = PCA(n_components=2)
reduced = pca.fit_transform(unreduced_feasonly)

sns.plot(data = 'reduced', hue = '')

reduceddf = pd.DataFrame(reduced, columns = ['A','B'])
df_to_plot = reduceddf
df_to_plot[feaset] = labels
sns.scatterplot(data=df_to_plot, x='A', y = 'B',hue=feaset)

kmeans = KMeans(n_clusters=2, random_state=0).fit(reduceddf)
sns.scatterplot(data=df_to_plot, x='A', y = 'B',hue=kmeans.labels_)

clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples= 10, gen_min_span_tree=True)
clusterer = clusterer.fit(reduceddf)
sns.scatterplot(data=df_to_plot, x='A', y = 'B',hue=clusterer.labels_)
