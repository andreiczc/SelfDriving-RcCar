import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
import tensorflow as tf
import csv
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import cv2


# noinspection DuplicatedCode
class AccuracyCallback(tf.keras.callbacks.Callback):

    def __init__(self, test_data, no_classes):
        super(AccuracyCallback, self).__init__()
        self.test_data = test_data
        self.no_classes = no_classes
        self.accuracy = 0

    def on_epoch_end(self, epoch, logs=None):
        correct = 0
        incorrect = 0

        x_test, y_actual = self.test_data

        y_predicted = self.model.predict(x_test, verbose=0)

        class_correct = [0] * self.no_classes
        class_incorrect = [0] * self.no_classes

        for i in range(len(y_predicted)):
            predicted_index = np.argmax(y_predicted[i])
            actual_index = np.argmax(y_actual[i])

            if predicted_index == actual_index:
                correct += 1
                class_correct[actual_index] += 1
            else:
                incorrect += 1
                class_incorrect[actual_index] += 1

        class_correct = np.array(class_correct)
        class_incorrect = np.array(class_incorrect)
        class_accuracy = class_correct / (class_correct + class_incorrect)
        print('Correct predictions: {}'.format(class_correct))
        print('Incorrect predictions: {}'.format(class_incorrect))
        print('Class Accuracy: {}'.format(class_accuracy))
        print('Model Accuracy: {}'.format(logs['accuracy']))
        print('Model Loss: {}'.format(logs['loss']))


def center_pixel_values(x):
    return (x - tf.math.reduce_mean(x)) / tf.math.reduce_std(x)


X = []
y = []

count = {'a': 0,
         'd': 0,
         'w': 0}

with open('../train_img/labels.csv') as file:
    reader = csv.reader(file)
    for line in reader:
        label = str(line[1])
        image = np.asarray(Image.open(line[0]))
        image = cv2.resize(image, (320, 180), interpolation=cv2.INTER_NEAREST)

        if label in ['a', 'd']:
            img_flipped = np.fliplr(image)
            if label == 'a':
                label_flipped = 'd'
            else:
                label_flipped = 'a'

            X.append(img_flipped)
            y.append(label_flipped)
            count[label_flipped] += 1

        X.append(image)
        y.append(label)
        count[label] += 1

label_binarizer = LabelBinarizer()
y_one_hot = label_binarizer.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42, stratify=y_one_hot)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

model = tf.keras.models.load_model('./models/nvidia_model')

history = model.fit(x=X_train, y=y_train, epochs=6, verbose=2, callbacks=[AccuracyCallback((X_train, y_train), 3)],
                    validation_split=0.15)

results = model.evaluate(X_test, y_test)
print('test loss, test acc: ', results)

model.save('./models/simple_model_flip')
