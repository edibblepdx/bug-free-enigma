# Ethan Dibble
# CNN for audio feature extraction
# Using the GTZAN dataset

import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Flatten, MaxPooling1D, Dense
from keras.saving import load_model
from keras import Model
import matplotlib.pyplot as plt
import argparse
import csv
import sys
import os

SAMPLE_RATE = 22050
N_MFCC = 40

class FeatureExtract:
    def __init__(self):
        pass

    def load_data(self, data_path, features_path):
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
                sys.stderr.write(f"Error processing {file_path}: {e}\n")

        return np.array(features), np.array(labels)

    def save_csv(self, path, features, labels):
        """write features to csv"""
        try:
            header = [f"mfcc{i}" for i in range(N_MFCC)]
            header.append("label")

            with open('features.csv', 'w', encoding='UTF8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

                for feature, label in zip(features, labels):
                    data = [a for a in feature]
                    data.append(label)
                    writer.writerow(data)

        except Exception as e:
            sys.stderr.write(f"Error writing to {path}: {e}\n")

    def load_csv(self, path):
        """load features from csv"""
        try:
            features = []
            labels = []

            data_features = pd.read_csv(path)
            feature_columns = data_features.columns[:-1]
            label_column = data_features.columns[-1]

            for _, row in data_features.iterrows():
                mfccs = row[feature_columns].values
                features.append(mfccs)
                labels.append(row[label_column])

        except Exception as e:
            sys.stderr.write(f"Error reading from {path}: {e}\n")

        return np.array(features), np.array(labels)

    def train(self, features, labels, predict=False):
        """train the cnn"""
        labels_onehot, label_encoder = self.__preprocess_labels(labels)
        self.model = self.__build_cnn(features, len(label_encoder.classes_))
        x_train, x_test, y_train, y_test = train_test_split(features, labels_onehot, test_size=0.2, random_state=42, stratify=labels_onehot)

        self.model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), verbose=1)
        self.feature_extractor = Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer(name="stop").output,
            #outputs=[layer.output for layer in self.model.layers[:-4]],
        )

        if predict:
            self.__predict(x_test, y_test, label_encoder)

    def extract(self, input):
        """extract features from input"""
        try:
            # https://keras.io/guides/sequential_model/#feature-extraction-with-a-sequential-model
            #print("Inputs shape:", np.shape(input))  # Debugging line
            return self.feature_extractor(input)

        except Exception as e:
            sys.stderr.write(f"Error extracting features: {e}\n")
            sys.stderr.write("train or load the model first\n")

    def save_model(self, path=None, overwrite=False):
        """save the cnn"""
        if path:
            self.model.save(path, overwrite=overwrite)

        else:
            self.model.save("./cnn.keras", overwrite=overwrite)

    def load_model(self, path):
        """load the cnn"""
        try:
            self.model = load_model(path)
            self.feature_extractor = Model(
                inputs=self.model.inputs,
                outputs=self.model.get_layer(name="stop").output,
                #outputs=[layer.output for layer in self.model.layers[:-4]],
            )

        except Exception as e:
            sys.stderr.write(f"Error loading model: {e}\n")

    def __predict(self, x_test, y_test, label_encoder):
        """testing only"""
        try:
            # predictions
            predictions = self.model.predict(x_test) # shape(num_inputs, num_classes)
            predicted_classes = np.argmax(predictions, axis=1) # largest indices
            predicted_labels = label_encoder.inverse_transform(predicted_classes)
            true_classes = np.argmax(y_test, axis=1)
            true_labels = label_encoder.inverse_transform(true_classes)

            # confusion matrix
            cm = confusion_matrix(true_labels, predicted_labels, labels=label_encoder.classes_)
            display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
            display.plot(cmap=plt.cm.Blues)
            plt.xticks(rotation=45)
            plt.title('Confusion Matrix')
            plt.show()

            # accuracy
            accuracy = np.sum(predicted_classes == true_classes) / len(true_classes)
            print (f"accuracy: {accuracy}")
        
        except Exception as e:
            sys.stderr.write(f"Error in prediction: {e}\n")

    def __preprocess_labels(self, labels):
        """one-hot encode categorical labels"""
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels) # returns an array of labels converted to integers
        labels_onehot = to_categorical(labels_encoded) # returns an array of one-hot encoded vector labels

        return labels_onehot, label_encoder

    def __build_cnn(self, features, num_classes):
        """Create a cnn model for feature extraction with input, convolutional, pooling layers"""
        input_shape = (features.shape[1], 1)
        model = Sequential([
            #Conv1D(64, 3, padding='same', activation='relu', input_shape=input_shape)
            #, MaxPooling1D(pool_size=2)
            #, Dropout(0.25)
            #, Conv1D(128, 3, padding='same', activation='relu')
            #, MaxPooling1D(pool_size=2)
            #, Dropout(0.25)
            #, Flatten()
            #, Dense(512, activation='relu')
            #, Dropout(0.5)
            #, Dense(num_classes, activation='softmax')
            Conv1D(64, 3, padding='same', activation='relu', input_shape=(features.shape[1], 1))
            , MaxPooling1D(pool_size=2)
            , Dropout(0.1)
            , Conv1D(128, 3, padding='same', activation='relu')
            , MaxPooling1D(pool_size=2)
            , Dropout(0.1)
            , Conv1D(256, 3, padding='same', activation='relu')
            , MaxPooling1D(pool_size=2)
            , Dropout(0.1)
            , Conv1D(512, 3, padding='same', activation='relu')
            , MaxPooling1D(pool_size=2)
            , Dropout(0.1)
            , Flatten()
            , Dense(2048, activation='relu')
            , Dense(1024, activation='relu')
            , Dense(256, activation='relu', name='stop')
            , Dense(num_classes, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

#def main(inputs, labels, output, save):
def main():
    """example use"""
    fe = FeatureExtract()
    features, labels = fe.load_data('Data/genres_original', 'Data/features_30_sec.csv')
    fe.save_csv('features.csv', features, labels)
    fe.train(features, labels, predict=True)
    fe.save_model(path='cnn6.keras', overwrite=True)

if __name__ == '__main__':
    """
    parser = argparse.ArgumentParser(
        prog='FeatureExtract.py'
        , description='Creates, Saves, Loads a CNN as a feature extractor for input to other models.'
    )
    parser.add_argument('-i', '--inputs', required=True, help='inputs path')
    parser.add_argument('-l', '--labels', required=True, help='labels path')
    parser.add_argument('-o', '--output', help='output name, default="cnn.keras"')
    parser.add_argument('-s', '--save', help='save model')
    args = parser.parse_args()
    main(args.inputs, args.labels, args.output, args.save)
    """
    main()