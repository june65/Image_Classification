import os
import glob
'''
# horses/humans 데이터셋 경로 지정
dog_breed_files = glob.glob('.\\data\\images\\images\\*')

name_array = [[0 for _ in range(2)] for _ in range(len(dog_breed_files))]
i = 0

dog_breed_image_names = [[0 for _ in range(1000)] for _ in range(len(dog_breed_files))]
content = ''
for filename in dog_breed_files:

    name_array = (filename.replace('.\\data\\images\\images\\n','')).split('-')
    dog_breed_image_names[i] = os.listdir(filename)


    content += name_array[0]+","+name_array[1]+"\n"  
    f = open("mix_breed.txt", 'wt', encoding='UTF8')
    f.write(content)  
    i += 1
'''

import tensorflow as tf

model = tf.keras.models.Sequential([
    # The first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    # Flatten
    tf.keras.layers.Flatten(),
    # 512 Neuron (Hidden layer)
    tf.keras.layers.Dense(64, activation='relu'),
    # 1 Output neuron
    tf.keras.layers.Dense(120, activation='sigmoid')
])

model.summary()

from keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
            optimizer=RMSprop(lr=0.001),
            metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1.0/255)

train_generator = train_datagen.flow_from_directory(
  './data/images/images',
  target_size=(100, 100),
  batch_size=128,
  class_mode='binary'
)
history = model.fit(
  train_generator,
  steps_per_epoch=5,
  epochs=10,
  verbose=120
)

model.save('./dog_breed_model')