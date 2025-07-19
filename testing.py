import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle

path_test = r'C:\Users\draco\.cache\kagglehub\datasets\msambare\fer2013\versions\1/test'
model = tf.keras.models.load_model('best_model.h5')  
test_datagen = ImageDataGenerator(rescale=1./255)

testing_set = test_datagen.flow_from_directory(
    path_test,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

Y_pred = model.predict(testing_set)
y_pred = np.argmax(Y_pred, axis=1)

classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

print("Classification Report:")
print(classification_report(testing_set.classes, y_pred, target_names=classes))
print("Confusion Matrix:")
print(confusion_matrix(testing_set.classes, y_pred))

with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
