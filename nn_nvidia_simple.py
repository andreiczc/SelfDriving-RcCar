import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

inputs = tf.keras.layers.Input(shape=(180, 320, 3))
lambda_1 = tf.keras.layers.Lambda(lambda x: (x - 128) / 128)(inputs)
convo_1 = tf.keras.layers.Conv2D(filters=24, kernel_size=5, strides=2, activation='relu')(lambda_1)
convo_2 = tf.keras.layers.Conv2D(filters=36, kernel_size=5, strides=2, activation='relu')(convo_1)
convo_3 = tf.keras.layers.Conv2D(filters=48, kernel_size=5, strides=2, activation='relu')(convo_2)
convo_4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(convo_3)
convo_5 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(convo_4)
flatten = tf.keras.layers.Flatten()(convo_5)
dense_1 = tf.keras.layers.Dense(100)(flatten)
dense_2 = tf.keras.layers.Dense(50)(dense_1)
dense_3 = tf.keras.layers.Dense(10)(dense_2)
predictions = tf.keras.layers.Dense(3, activation='softmax')(dense_3)

model = tf.keras.models.Model(inputs=inputs, outputs=predictions, name='nvidia_model')

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.save('models/nvidia_model')
