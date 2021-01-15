import keras
import numpy as np
from PIL import Image

#pomocna metoda slouzici pro upravu obrazku a naslednou predikci melanomu
def predict_melanon(file_name):
    model = keras.models.load_model('network\melanomNN.h5', compile = True)

    image = Image.open(file_name)
    image = image.resize((64,64))

    #print(image.format)
    #print(image.mode)
    #print(image.size)

    images = np.asarray(image)
    images = images.astype('float32')
    images = images / 255
    images = np.reshape(images,[1,64,64,3])

    #print(images.shape)
    prediction = model.predict(images)

    return prediction