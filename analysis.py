#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
from helper import *

def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

nRowsRead = 10000
df1 = pd.read_csv('Chicago_Crimes_2012_to_2017.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Chicago_Crimes_2012_to_2017.csv'
nRow, nCol = df1.shape
print("Total:" + str(nRow) + " rows and " + str(nCol) + " columns")
#print(df1.head())
for col in df1.columns:
    print(col)
print("Sample geo-info, X Coordinate:")
print(df1["X Coordinate"].iloc[0])
print("Sample geo-info, latitude:")
print(df1["Latitude"].iloc[0])
print("Sample time-info, time series:")
time_str = df1["Date"].iloc[0]
print(convert_timestr_to_epoch(time_str))

#plotCorrelationMatrix(df1, 8)

df2 = df1[["Latitude", "Longitude", "Date"]]
df2.dropna()
df4 = df2[["Latitude", "Longitude"]]
df4['epoch'] = df1.apply(lambda row: convert_timestr_to_epoch(row["Date"]), axis=1)
df4['hrs'] = df1.apply(lambda row: convert_timestr_to_hours(row["Date"]), axis=1)
print("Before normalization")
print(df4)
print("After normalization")
df5=(df4-df4.mean())/df4.std() # normalization
print(df5)
df5.dropna()
df6 = clean_dataset(df5)
df5 = df5[["Latitude", "Longitude"]]
df5 = df5.head(5000)
reduced_data = df5.values
kmeans = KMeans(n_clusters=31, random_state=0).fit(reduced_data)

h = 0.02
# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    cmap=plt.cm.Paired,
    aspect="auto",
    origin="lower",
)

plt.plot(reduced_data[:, 0], reduced_data[:, 1], "k.", markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=20,
    linewidths=1,
    color="w",
    zorder=10,
)
plt.title(
    "K-means clustering on the crime-related geographical data\n"
    "Centroids are marked with white cross"
)
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
