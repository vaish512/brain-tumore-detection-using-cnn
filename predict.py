import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load Model
model = tf.keras.models.load_model('brain_tumor_cnn_model.h5')

# Load Image
img_path = 'sample_mri.jpg'

img = image.load_img(img_path, target_size=(128,128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("Tumor Detected")
else:
    print("No Tumor Detected")
