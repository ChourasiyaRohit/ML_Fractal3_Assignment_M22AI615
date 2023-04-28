import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('Handwritten_TrainImage_NN.model')

# Define the path to the validation images
val_dir = 'val'

# Loop through each folder (0-9) in the validation directory
for i in range(10):
    folder_path = os.path.join(val_dir, str(i))
    print(f"Processing folder: {folder_path}")

    # Loop through each image in the folder
    for j, file_name in enumerate(os.listdir(folder_path)):
        try:
            # Load the image and convert it to grayscale
            img = cv2.imread(os.path.join(folder_path, file_name))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize the image to 32x32 pixels and normalize the pixel values
            img_resized = cv2.resize(img_gray, (32, 32))
            img_normalized = img_resized / 255.0

            # Reshape the image to match the input shape of the model
            img_reshaped = img_normalized.reshape(-1, 32, 32, 1)

            # Make a prediction using the model
            prediction = model.predict(img_reshaped)

            # Print the predicted class and confidence score
            class_pred = np.argmax(prediction)
            confidence = prediction[0][class_pred]
            print(f"File {file_name}: Predicted class = {class_pred}, Confidence = {confidence}")

            # Display the image and pause for 1 second
            plt.imshow(img_resized, cmap=plt.cm.binary)
            plt.title(f"Predicted class: {class_pred}")
            plt.show(block=False)
            plt.pause(1)
            plt.close()

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    print("")
