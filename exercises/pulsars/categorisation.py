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
Y = htru2.data.targets

# # variable information
# print(htru2.variables)

# sns.pairplot(data=X, corner=True, kind="kde")
# plt.show()


# try out the old k means, see what happens:
def try_out_k_means():
    def df_standardiser(df: pd.DataFrame) -> pd.DataFrame:
        standardised_df = pd.DataFrame()
        for col in df.columns:
            standardised_df[col] = (df[col] - df[col].mean()) / df[col].std()
        return standardised_df

    # sns.pairplot(data=df_standardiser(X), corner=True, kind="kde")
    # plt.show()

    standardised_X = df_standardiser(X)

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(standardised_X)
    labels = kmeans.fit_predict(standardised_X)

    def percentage_correct_calculator(found_labels, correct_labels):
        num_correct = np.sum(np.logical_and(found_labels, correct_labels))
        return (num_correct / len(found_labels)) * 100

    print(
        f"Number in category 0: {len(labels) - np.sum(labels)}. \nNumber in Category 1: {np.sum(labels)}."
    )
    print(
        f"Percentage correct: {percentage_correct_calculator(found_labels=labels, correct_labels=Y.to_numpy()[:, 0]):.2f}%"
    )

    unique_labels = np.unique(labels)
    for i in unique_labels:
        plt.scatter(
            standardised_X.to_numpy()[labels == i, 0],
            standardised_X.to_numpy()[labels == i, 1],
            label=i,
        )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    try_out_k_means()
