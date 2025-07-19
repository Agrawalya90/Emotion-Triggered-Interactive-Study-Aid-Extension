import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np

path_train = r'C:\Users\draco\.cache\kagglehub\datasets\msambare\fer2013\versions\1/train'
path_test = r'C:\Users\draco\.cache\kagglehub\datasets\msambare\fer2013\versions\1/test'

train_datageb = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    rotation_range=10,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True
)

test_datgen = ImageDataGenerator(rescale=1./255)

training_set = train_datageb.flow_from_directory(
    path_train,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)

testing_set = test_datgen.flow_from_directory(
    path_test,
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

cnn = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])

cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)

history = cnn.fit(
    training_set,
    validation_data=testing_set,
    epochs=25,
    callbacks=[early_stop, checkpoint]
)

cnn.save('model.h5')


