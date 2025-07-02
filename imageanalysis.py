import tensorflow as tf
import lz4
import os
import cv2
import numpy as np
from google.colab import drive
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from numpy import asarray
from PIL import Image
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN # if you encounter the error 'No module named mtcnn', execute the following command !pip install mtcnn

# Set parameters
IMAGE_SIZE = (128, 128)  # Resize images to 128x128
BATCH_SIZE = 32
EPOCHS = 100
drive.mount('/content/drive')
DATA_DIR = "/content/drive/MyDrive/image_classification_dataset"

# draw an image with detected objects
def draw_image_with_boxes(filename, result_list):
  # load the image
  data = pyplot.imread(filename)
  # plot the image
  pyplot.imshow(data)
  # get the context for drawing boxes
  ax = pyplot.gca()
  # plot each box
  for result in result_list:
    # get coordinates
    x, y, width, height = result['box']
    # create the shape
    rect = Rectangle((x, y), width, height, fill=False, color='red')
    # draw the box
    ax.add_patch(rect)

  # draw the dots (for eyes, nose, and mouth)
  for key, value in result['keypoints'].items():
    # create and draw dot
    dot = Circle(value, radius=2, color='red')
    ax.add_patch(dot)

# Source: https://www.sitepoint.com/keras-face-detection-recognition/
def extract_face_from_image(image_path, required_size=(128, 128)):
  # load image and detect faces
  image = pyplot.imread(image_path)
  detector = MTCNN()
  faces = detector.detect_faces(image)

  face_images = []

  for face in faces:
    # extract the bounding box from the requested face
    x1, y1, width, height = face['box']
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face_boundary = image[y1:y2, x1:x2]

    # resize pixels to the model size
    face_image = Image.fromarray(face_boundary)
    face_image = face_image.resize(required_size)
    face_array = asarray(face_image)
    face_images.append(face_array)

  return face_images

# Load and preprocess data
def load_data(data_dir):
    image_data = []
    labels = []
    class_names = os.listdir(data_dir)

    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)



            # img = tf.keras.utils.load_img(img_path, target_size=IMAGE_SIZE)

            # load image from file
            pixels = pyplot.imread(img_path)
            # create the detector, using default weights
            detector = MTCNN()
            # detect faces in the image
            faces = detector.detect_faces(pixels)
            # display faces on the original image
            draw_image_with_boxes(img_path, faces)
            extracted_face = extract_face_from_image(img_path)

            img_array = tf.keras.utils.img_to_array(cv2.resize(extracted_face[0],(128,128))) #get the first image. previously just img
            image_data.append(img_array)
            labels.append(idx)

    image_data = tf.convert_to_tensor(image_data) / 255.0  # Normalize images

    labels = tf.convert_to_tensor(labels)
    return image_data, labels, class_names

print("Loading data...")
image_data, labels, class_names = load_data(DATA_DIR)
print(f"Classes: {class_names}")

# if you encounter the error, LZ4 is not installed. Install it with pip: https://python-lz4.readthedocs.io/, !pip install LZ4
labels
image_data[0]
image_data[1]
image_data[2]
image_data[3]
image_data[4]

# Split into train and validation sets
# Convert TensorFlow tensors to NumPy arrays before using train_test_split
image_data_np = image_data.numpy()
labels_np = labels.numpy()

# Split into train and validation sets
# X_train, X_val, y_train, y_val = train_test_split(image_data, labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(image_data_np, labels_np, test_size=0.2, random_state=42)

# Define the CNN model
def create_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names), activation='softmax')
    ])
    return model

model = create_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=EPOCHS
)

# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.2f}")

# Save the model
# Save the model
model.save("person_classifier_model.h5")
print("✅ Model saved as person_classifier_model.h5")

pyplot.imshow(image_data_np[1])
image_array = np.expand_dims(image_data_np[1], axis=0)
prediction = model.predict(image_array)
print("Predicted class:", class_names[np.argmax(prediction)])