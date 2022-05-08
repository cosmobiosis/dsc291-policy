#from mpl_toolkits.mplot3d import Axes3D
#from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans

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
#plotCorrelationMatrix(df1, 8)

df2 = df1[["Latitude", "Longitude"]]
df2.dropna()
mat = clean_dataset(df2).values
kmeans = KMeans(n_clusters=55, random_state=0).fit(mat)


