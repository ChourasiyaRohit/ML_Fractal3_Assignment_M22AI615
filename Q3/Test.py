import os
import cv2
import numpy as np
import tensorflow as tf

IMG_WIDTH = 128
IMG_HEIGHT = 128

model = tf.keras.models.load_model("chart_classifier_model.h5")

def preprocess_image(image):
    # Resize the image to match the input size of the model
    img = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

    # Convert the image to a 3D array of floats and rescale the pixel values to [0, 1]
    img = np.array(img, dtype=np.float32) / 255.0

    # Add a batch dimension to the image array
    img = np.expand_dims(img, axis=0)

    return img

test_path = "test"
test_files = os.listdir(test_path)
print("Number of test files:", len(test_files))

for filename in test_files:
    if not filename.endswith(".png"):
        continue

    image_path = os.path.join(test_path, filename)
    input_image = cv2.imread(image_path)

    if input_image is None:
        print("Failed to read image:", image_path)
        continue

    input_image = preprocess_image(input_image)

    predictions = model.predict(input_image)

    class_names = ["bar_chart", "line_chart", "pie_chart", "scatter_plot", "table"]
    predicted_class_index = np.argmax(predictions)
    predicted_class_name = class_names[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    print("Image name:", filename)
    print("Predicted class:", predicted_class_name)
    print("Confidence: {:.2f}".format(confidence))
    print()
