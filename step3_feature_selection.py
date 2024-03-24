import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    dataset = np.load("./feature.npz")

    person_id = dataset["person_id"]
    x = dataset['x']
    y = dataset['y']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=22)

    ## Control Parameter
    N = 40  # Find The Number of Selected Features

    train_data = np.column_stack((x_train, y_train))
    df = pd.DataFrame(train_data)
    abs_corr = np.absolute(df.corr().iloc[:-1, 82])
    idx = np.argpartition(abs_corr, -N)[-N:]

    x_train = df[idx].to_numpy()
    x_test = pd.DataFrame(x_test)[idx].to_numpy()

    # colormap = plt.cm.gist_heat
    # plt.figure(figsize=(12, 12))
    # sns.heatmap(df.corr(), linewidths=0.1, vmax=0.5, cmap=colormap, linecolor="white", annot=True)
    # plt.show()

    plt.figure()
    plt.bar(range(abs_corr.shape[0]), abs_corr, color="red")
    plt.xlabel("Feature Index")
    plt.ylabel("Absolute Value of Pearson Correlation Coefficient")
    plt.ylim([0, 1])
    plt.grid(True)
    plt.show()