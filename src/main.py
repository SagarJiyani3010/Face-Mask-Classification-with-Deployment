import numpy as np
import pandas as pd 
import tensorflow as tf 
from keras.preprocessing.image import ImageDataGenerator
import os

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.99 :
            print("\nYou reached to 99% acuuracy, so training has been stopped...!")
            self.model.stop_training =True

dataset = '../input/'
train = os.path.join(dataset, 'Train/')
test = os.path.join(dataset, 'Test/')
validation = os.path.join(dataset, 'Validation/')

callbacks = myCallback()

model = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(100,100,3)),
                                    tf.keras.layers.MaxPooling2D(2,2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(246, activation='relu'),
                                    tf.keras.layers.Dense(1, activation='sigmoid')])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale=1.0/255.0)
train_generator = train_datagen.flow_from_directory(
    train,
    batch_size=256,
    class_mode='binary',    
    target_size=(100,100)
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test,
    batch_size=256,
    class_mode='binary',
    target_size=(100,100)
)

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
valid_generator = valid_datagen.flow_from_directory(
    validation,
    batch_size=256,
    class_mode='binary',
    target_size=(100,100)
)

model.fit(train_generator, epochs=5, validation_data=valid_generator, callbacks = [callbacks])
model.evaluate(test_generator)

model.save('../deployment/models/model.h5')