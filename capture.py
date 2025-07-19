import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import cv2
import time
import datetime
import pyautogui
import subprocess


model = tf.keras.models.load_model('model.h5')
cap = cv2.VideoCapture(0)

total_captures = 2
interval_seconds = 2
classes =['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

for i in range(total_captures):
    ret, frame = cap.read()

    img_path = f"minute_capture_{i+1}.jpg"
    cv2.imwrite(img_path, frame)
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    pred = model.predict(img_array)
    pred_class_index = np.argmax(pred, axis=1)[0]
    predicted_label = classes[pred_class_index]
    print(predicted_label)

    if predicted_label not in ['happy', 'neutral']:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        screenshot = pyautogui.screenshot()
        screenshot.save(f'screenshot_{timestamp}.png')
        path_ss=f'screenshot_{timestamp}.png'
        subprocess.run(["python", "ocr.py",path_ss])


    time.sleep(interval_seconds)

cap.release()
cv2.destroyAllWindows()
print("✅ Finished capturing 5 images over 5 minutes.")
