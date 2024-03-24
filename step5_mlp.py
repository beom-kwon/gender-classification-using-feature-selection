import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
from keras import Sequential, Input
from keras.callbacks import EarlyStopping
from keras.layers import Dense


def BuildMLP(N):
    # Title: Gender and body mass index classification using a Microsoft Kinect sensor
    # Author: Andersson et al.
    # Year: 2015
    # The MLP was set to have a single hidden layer with 4 hidden units.
    # Sigmoidal activation function was used for all units.
    model = Sequential()
    model.add(Input(shape=(N,)))
    model.add(Dense(4, activation="sigmoid"))
    model.add(Dense(1, activation="sigmoid"))
    return model


if __name__ == "__main__":
    dataset = np.load("./feature.npz")

    person_id = dataset["person_id"]
    x = dataset['x']
    y = dataset['y']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=22)

    # Control Parameter
    N = 70 # Find The Number of Selected Features

    train_data = np.column_stack((x_train, y_train))
    df = pd.DataFrame(train_data)
    abs_corr = np.absolute(df.corr().iloc[:-1, 82])
    idx = np.argpartition(abs_corr, -N)[-N:]

    x_train = df[idx].to_numpy()
    x_test = pd.DataFrame(x_test)[idx].to_numpy()

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train,
                                                      random_state=22)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = scaler.transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    # print(x_train_scaled.shape, x_val_scaled.shape, x_test_scaled.shape)

    model = BuildMLP(N)

    Epoch = 1000

    model.compile(optimizer="adam",
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    es_cb = EarlyStopping(patience=2, restore_best_weights=True)
    callback = [es_cb]

    h = model.fit(x_train_scaled, y_train, epochs=Epoch,
                  callbacks=callback, verbose=0,
                  validation_data=(x_val_scaled, y_val))

    y_pred = tf.round(model.predict(x_test_scaled, verbose=0))

    tp, fn, fp, tn = confusion_matrix(y_test, y_pred).ravel()
    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Female", "Male"], cmap=plt.cm.Blues)

    print(">> ACC: %.4f" % ((tp + tn) / (tp + fn + fp + tn)))
    print(">> TPR: %.4f" % (tp / (tp + fn)))
    print(">> TNR: %.4f" % (tn / (tn + fp)))

    plt.figure()
    plt.plot(pd.DataFrame(h.history)["loss"], label="Traininig")
    plt.plot(pd.DataFrame(h.history)["val_loss"], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.legend()
    plt.show()