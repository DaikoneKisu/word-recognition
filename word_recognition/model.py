"""main file of model"""

# @title Import relevant modules
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib as plt
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# The following lines adjust the granularity of reporting.
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

# The following line improves formatting when ouputting NumPy arrays.
np.set_printoptions(linewidth=200)

AMOUNT_OF_LETTER_CATEGORIES = 27  # Numbers of letters
amount_of_brightness = 255
# @title Hyperparameters
learning_rate = 0.002
batch_size = 256
validation_split = 0.3


def train():
    """train the model"""
    # @title Reading the training/test datasets from csv files

    train = None

    try:
        train = pd.read_csv("word_recognition/assets/emnist-letters-train.csv")
    except FileNotFoundError:
        raise Exception("Error: No se encontró el archivo CSV requerido")

    # @title Dividing the training/test labels & features
    # Training Set
    train_labels = np.array(train.iloc[:, 0].values)  # Array with results [1,2,3,4....]
    train_features = np.array(
        train.iloc[:, 1:].values
    )  # Array with indexed-pixels [...](784 pixels)

    # train_features.shape -> [rows number, columns number]
    train_features_amount = train_features.shape[0]  # Records (= 88799)

    # Convert 784px to bidimentional
    train_features = np.reshape(train_features, (train_features_amount, 28, 28))

    # @title Normalization and Reshaping of the features
    train_features = train_features / amount_of_brightness

    # Convert the labels to One-hot encoding labels
    # [1,1,2] -> [[1,0,...],[1,0,...],[0,1...]]
    train_labels = keras.utils.to_categorical(train_labels, AMOUNT_OF_LETTER_CATEGORIES)

    train_x, train_y = train_features, train_labels

    # @title Define the plotting function
    def plot_curve(epochs, hist, list_of_metrics):
        """Plot a curve of one or more classification metrics vs. epoch."""
        # list_of_metrics should be one of the names shown in:
        # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#define_the_model_and_metrics

        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Value")

        for m in list_of_metrics:
            x = hist[m]
            plt.plot(epochs[1:], x[1:], label=m)

        plt.legend()
        #plt.show()
        plt.savefig("plot-train.jpg")

        print("Loaded the plot_curve function.")

    # @title Neural Network Architecture definition
    def create_model(input_learning_rate):
        image_pixels = 28

        model = keras.models.Sequential()

        model.add(keras.layers.Flatten(input_shape=(image_pixels, image_pixels)))
        model.add(keras.layers.Dense(units=256, activation="relu"))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.Dense(units=128, activation="relu"))
        model.add(keras.layers.Dropout(rate=0.1))
        model.add(keras.layers.Dense(units=64, activation="relu"))
        model.add(keras.layers.Dropout(rate=0.1))
        model.add(
            keras.layers.Dense(units=AMOUNT_OF_LETTER_CATEGORIES, activation="softmax")
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=input_learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def train_model(
        model,
        train_features,
        train_label,
        epochs,
        batch_size_func=None,
        validation_split_func=0.1,
    ):
        history = model.fit(
            x=train_features,
            y=train_label,
            batch_size=batch_size_func,
            epochs=epochs,
            shuffle=True,
            validation_split=validation_split_func,
        )

        epochs = history.epoch
        hist = pd.DataFrame(history.history)

        return epochs, hist

    # @title Model
    # Establish the model's topography.

    my_model = create_model(learning_rate)

    epochs = 30

    # Train the model on the normalized training set.
    epochs, hist = train_model(
        my_model, train_x, train_y, epochs, batch_size, validation_split
    )

    # Plot a graph of the metric vs. epochs.
    list_of_metrics_to_plot = ["accuracy"]
    plot_curve(epochs, hist, list_of_metrics_to_plot)

    my_model.save("model.keras")


def test():
    """test the model"""

    my_model = keras.saving.load_model("model.keras", compile=True)

    if my_model is None:
        raise Exception("There is no model saved")

    test = None

    try:
        test = pd.read_csv("word_recognition/assets/emnist-letters-test.csv")
    except FileNotFoundError:
        print("Error: No se encontró el archivo CSV requerido")

    # Test Set
    test_labels = np.array(test.iloc[:, 0].values)
    test_features = np.array(test.iloc[:, 1:].values)

    test_features_amount = test_features.shape[0]  # Records (> 14000)

    test_features = np.reshape(test_features, (test_features_amount, 28, 28))

    test_features = test_features / amount_of_brightness

    test_labels = keras.utils.to_categorical(test_labels, AMOUNT_OF_LETTER_CATEGORIES)

    test_x, test_y = test_features, test_labels

    # Evaluate against the test set.
    print("\n Evaluate the new model against the test set:")
    my_model.evaluate(x=test_x, y=test_y, batch_size=batch_size)
