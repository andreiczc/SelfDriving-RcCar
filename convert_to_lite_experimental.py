import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

converter = tf.compat.v1.lite.TFLiteConverter.from_saved_model('./models/simple_model_flip')
converter.inference_type = tf.compat.v1.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}
tflite_model = converter.convert()

with tf.io.gfile.GFile('simple_model.tflite', 'wb') as f:
    f.write(tflite_model)

print('model saved')
