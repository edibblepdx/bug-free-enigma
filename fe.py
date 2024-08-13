import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Flatten, MaxPooling1D, Dense
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

    for index, row in data_features.iterrows():
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
        print(mfccs.shape)
        print(mfccs_scaled.shape)
        features.append(mfccs_scaled)
        labels.append(row['label'])

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

    return labels_onehot, le

def main():
    """example use"""
    features, labels = load_data('GTZAN/Data/genres_original', 'GTZAN/Data/features_30_sec.csv')
    features = features.reshape(features.shape[0], features.shape[1], 1) #!
    cnn_model = create_cnn()

    extracted_features = cnn_model.predict(features)

    """
    # example with a dense neural network classifier
    dnn_model = Sequential([
        Dense(512, activation='relu')
        , Dropout(0.5)
        , Dense(len(le.classes_), activation='softmax')
    ])
    # https://keras.io/api/optimizers/
    dnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    labels_onehot, label_encoder = preprocess_labels(labels)
    x_train, x_test, y_train, y_test = train_test_split(extracted_features, labels_onehot, test_size=0.2, random_state=42, stratify=labels_onehot)

    history = dnn_model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=1)
    print(history.history)

    predictions = dnn_model.predict(x_test)
    """
    print(extracted_features.shape)

if __name__ == '__main__':
    main()