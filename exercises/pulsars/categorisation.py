import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fcmeans import FCM
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from ucimlrepo import fetch_ucirepo

# fetch dataset
htru2 = fetch_ucirepo(id=372)

# data (as pandas dataframes)
X = htru2.data.features
y = htru2.data.targets

# # variable information
# print(htru2.variables)

# sns.pairplot(data=X, corner=True, kind="kde")
# plt.show()
#


def df_standardiser(df: pd.DataFrame) -> pd.DataFrame:
    standardised_df = pd.DataFrame()
    for col in df.columns:
        standardised_df[col] = (df[col] - df[col].mean()) / df[col].std()
    return standardised_df


# sns.pairplot(data=df_standardiser(X), corner=True, kind="kde")
# plt.show()

X = df_standardiser(X)

kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
print(kmeans.labels_)
labels = kmeans.fit_predict(X)
print(labels)
unique_labels = np.unique(labels)

# for i in unique_labels:
#     plt.scatter(X[labels == i, 0], X[labels == i, 1], label=i)
# plt.show()
#
#
#
#
# c_values = [2]
# ss_values = []
# ss_stds = []
# for c in c_values:
#     ss_list = []
#
#     for i in range(100):
#         fcm = FCM(n_clusters=c, m=2.0)
#         fcm.fit(X)
#         fcm_centers = fcm.centers
#         fcm_labels = fcm.predict(X)
#         fcm_soft_labels = fcm.soft_predict(X)
#
#         ss_scores = silhouette_samples(X, fcm_labels)
#         ss = np.mean(ss_scores)
#         ss_list.append(ss)
#     ss_mean = np.mean(ss_list)
#     ss_std = np.std(ss_list)
#     print("Mean ss scores : {:}, for c = {:}".format(ss_mean, c))
#     ss_values.append(ss_mean)
#     ss_stds.append(ss_std)
# plt.bar(c_values, ss_values, yerr=ss_stds)
# plt.title("Silhouette scores for FCM clustering")
# plt.xlabel("number of clusters used, c")
# plt.ylabel("mean silhouette score")
# plt.show()
