import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image

model = keras.models.load_model('./horse_vs_human')

path = 'C:/Users/82108/Desktop/horse/horse.jpg'

img=tf.keras.preprocessing.image.load_img(path,target_size=(300, 300))

x=tf.keras.preprocessing.image.img_to_array(img)
x=np.expand_dims(x, axis=0)
images = np.vstack([x])

classes = model.predict(images, batch_size=10)

print(classes[0])