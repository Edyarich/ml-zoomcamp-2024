import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import tensorflow.lite as tflite

import numpy as np

from io import BytesIO
from urllib import request
from PIL import Image


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size, rescale):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    arr = np.array(img, dtype='float32')

    arr *= rescale
    return arr


model = load_model('model_2024_hairstyle.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model_2024_hairstyle.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


interpreter = tflite.Interpreter(model_path='model_2024_hairstyle.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

img = download_image('https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg')
img_processed = prepare_image(img, target_size=(200, 200), rescale=1.0/255)
inp = np.array([img_processed])
interpreter.set_tensor(input_index, inp)
interpreter.invoke()

preds = interpreter.get_tensor(output_index)
