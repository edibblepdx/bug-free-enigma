# Ethan Dibble
# CNN for audio feature extraction

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Flatten, MaxPooling1D, Dense
import matplotlib.pyplot as plt
import os

SAMPLE_RATE = 22050
N_MFCC = 40

def load_data(data_path, features_path):
    """
    data_path should be something like Data/genres_original
    features_path should be something like Data/features_30_sec.csv
    """
    features = []
    labels = []

    data_features = pd.read_csv(features_path)

    for _, row in data_features.iterrows():
        try:
            file_path = os.path.join(data_path, f"{row['label']}", f"{row['filename']}")

            # load the audio file
            audio, sample_rate = librosa.load(file_path, sr=SAMPLE_RATE)

            # extract MFCC (Mel Frequency Capstone Coefficients) features
            mfccs = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=N_MFCC) # returns np.ndarray [shape=(…, n_mfcc, t)]
            mfccs_scaled = np.mean(mfccs, axis=1) # calculate the mean over the time frames

            # mfccs are calculated using the discrete cosine transform of the list of mel log powers { DCT of log(abs(FT)) }
            # mfccs are ordered in increasing frequency, over time frames
            # scaling the mfccs gives a vector of same length for each input that may have varying time frames
            # should also simplify features

            # append features and labels
            features.append(mfccs_scaled)
            labels.append(row['label'])
        
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return np.array(features), np.array(labels)

def create_cnn(inputs):
    """Create a cnn model for feature extraction with input, convolutional, pooling layers"""
    input_shape = (inputs.shape[1], 1)
    cnn_model = Sequential([
        Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape)
        , MaxPooling1D(pool_size=2)
        , Dropout(0.25)
        , Conv1D(128, 3, padding='same', activation='relu')
        , MaxPooling1D(pool_size=2)
        , Dropout(0.25)
        , Flatten()     # flatten to 1-D for input into training layers
    ])

    return cnn_model

def preprocess_labels(labels):
    """one-hot encode categorical labels"""
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels) # returns an array of labels converted to integers
    labels_onehot = to_categorical(labels_encoded) # returns an array of one-hot encoded vector labels

    return labels_onehot, label_encoder

def main():
    """example use"""
    features, labels = load_data('GTZAN/Data/genres_original', 'GTZAN/Data/features_30_sec.csv')
    features = features.reshape(features.shape[0], features.shape[1], 1) #!
    cnn_model = create_cnn(features)

    extracted_features = cnn_model.predict(features)

    # what follows is a training example with a dense neural network

    # one-hot labels and splitting sets
    labels_onehot, label_encoder = preprocess_labels(labels)
    x_train, x_test, y_train, y_test = train_test_split(extracted_features, labels_onehot, test_size=0.2, random_state=42, stratify=labels_onehot)

    # create model
    """
    dnn_model = Sequential([
        Dense(512, activation='relu', input_shape=(x_train.shape[1],))
        , Dropout(0.5)
        , Dense(len(label_encoder.classes_), activation='softmax')
    ])
    """
    dnn_model = Sequential([
        Dense(256, activation='relu', input_shape=(x_train.shape[1],))
        , Dense(128, activation='relu')
        , Dense(64, activation='relu')
        , Dense(10, activation='softmax')
    ])
    # https://keras.io/api/optimizers/
    dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # training
    # history = dnn_model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test), verbose=1)
    history = dnn_model.fit(x_train, y_train, epochs=20, batch_size=128)
    print(history.history)

    # predictions
    predictions = dnn_model.predict(x_test) # shape(num_inputs, num_classes)
    predicted_classes = np.argmax(predictions, axis=1) # largest indices
    predicted_labels = label_encoder.inverse_transform(predicted_classes)
    true_classes = np.argmax(y_test, axis=1)
    true_labels = label_encoder.inverse_transform(true_classes)

    # confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=label_encoder.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    display.plot(cmap=plt.cm.Blues)
    plt.xticks(rotation=45)
    plt.title('Confusion Matrix DNN')
    plt.show()

    # accuracy
    accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
    print (f"accuracy: {accuracy}")

if __name__ == '__main__':
    main()