import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fcmeans import FCM
from pandas.core.common import random_state
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from ucimlrepo import fetch_ucirepo

# fetch dataset
htru2 = fetch_ucirepo(id=372)

# data (as pa_ndas dataframes)
X = htru2.data.features
Y = htru2.data.targets

# # variable information
# print(htru2.variables)

# sns.pairplot(data=X, corner=True, kind="kde")
# plt.show()


TRYING_OUT_K_MEANS = False


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
    if TRYING_OUT_K_MEANS:
        try_out_k_means()

    # it appears that k means isn't our lad, let's try and be fancy
    # and build a classifier

    # Things to do:
    # we need to split the data into train and test
    #   - this might mean we need to scramble the order of the data to make sure
    #     that the pulsars are evenly distributed. Let's cheat and plot the pulsar locations
    #     before and after shuffling

    rng = np.random.default_rng(seed=42)
    y = Y.to_numpy()[:, 0]
    shuffled_y = rng.permutation(y)
    # check we have the same number of 1s after the shuffle
    assert np.sum(y) == np.sum(shuffled_y)

    # plot them to see how uniform it is
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].plot(y)
    axs[1].plot(shuffled_y)
    # plt.show()

    # this plot shows that shuffling makes it much more uniform
    # so we'll do that to the entire dataset
    # we need to join X and Y together so that we keep the labels sane
    # and then split them again afterwards

    def entire_dataset_shuffler(data, targets):
        # concat the data together
        big_df = pd.concat([data, targets], axis=1)
        print(big_df.shape)
        # using sklearn.utils.shuffle
        shuffled_df = shuffle(big_df, random_state=42)
        # shuffled_targets = shuffled_df["class"]
        # shuffled_data = shuffled_df.drop(columns=["class"])
        return shuffled_df

    shuffled_dataset = entire_dataset_shuffler(X, Y)

    # now we can split into test and training datasets
    # my research tells me that sklearn can just do all this (including shuffling) for me (of course it can...)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.25, random_state=42, shuffle=True
    )
    # I'm giving myself points for working out I needed to shuffle it. Even though if
    # I had done any reading at all I'd have saved half an hour

    # we're gonna try a support vector machine, as I play support classes and like linear algebra

    from sklearn import svm

    classifier = svm.SVC()
    classifier.fit(X_train, Y_train)

    # now we see how well it did
    import sklearn.metrics as skmet

    Y_predicted = classifier.predict(X_test)

    accuracy = skmet.accuracy_score(Y_test, Y_predicted)
    precision = skmet.precision_score(Y_test, Y_predicted)
    recall = skmet.recall_score(Y_test, Y_predicted)
    f1 = skmet.f1_score(Y_test, Y_predicted)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
