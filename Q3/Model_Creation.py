import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define constants
IMG_WIDTH = 128
IMG_HEIGHT = 128
BATCH_SIZE = 32
NUM_EPOCHS = 10

# Load train_val.csv file containing image filenames and their corresponding labels
train_df = pd.read_csv("charts/train_val.csv")

# Convert the "image_index" column to strings
train_df["image_index"] = train_df["image_index"].astype(str)

# Create ImageDataGenerator for data augmentation and rescaling
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% for training, 20% for validation
)

# Load train and validation images using ImageDataGenerator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="charts/train_val",
    x_col="image_index",
    y_col="type",
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    subset="training"
)

val_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory="charts/train_val",
    x_col="image_index",
    y_col="type",
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    subset="validation"
)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(5, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, epochs=NUM_EPOCHS, validation_data=val_generator)

# Save the trained model
model.save("chart_classifier_model.h5")
