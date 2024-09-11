import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import Xception
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D
dataset_dir = 'archive'
img_width, img_height = 224, 224
batch_size = 32
epochs = 5
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    zoom_range=0.2,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.5, 1.5],
)
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
base_model = Xception(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(Dense(3, activation='softmax'))  # 3 classes: Rock, Paper, Scissors
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs)
model.save_weights('gesture_model_weights_xc_5e.h5')