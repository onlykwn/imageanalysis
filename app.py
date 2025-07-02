from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
import os

app = Flask(__name__)
app.secret_key = "something-secret"

# Load model and face detector
model = load_model("person_classifier_model.h5")
detector = MTCNN()
class_names = sorted(os.listdir("image_classification_dataset"))

@app.route('/')
def home():
    prediction = session.pop('prediction', None)
    face_path = session.pop('face_path', None)
    status = session.pop('status', None)
    retrain_status = session.pop('retrain_status', None)
    return render_template('index.html', prediction=prediction, face_path=face_path, status=status, retrain_status=retrain_status)

@app.route('/add_class', methods=['POST'])
def add_class():
    class_name = request.form['class_name'].strip()
    uploaded_files = request.files.getlist('files')

    save_dir = os.path.join("image_classification_dataset", class_name)
    os.makedirs(save_dir, exist_ok=True)

    for file in uploaded_files:
        if file:
            filename = file.filename.replace(" ", "_")
            file_path = os.path.join(save_dir, filename)

            # If file exists, make it unique
            counter = 1
            base, ext = os.path.splitext(filename)
            while os.path.exists(file_path):
                filename = f"{base}_{counter}{ext}"
                file_path = os.path.join(save_dir, filename)
                counter += 1

            file.save(file_path)

    session['status'] = f"{len(uploaded_files)} image(s) saved to class '{class_name}'."
    return redirect(url_for('home'))

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if not file:
            session['prediction'] = "NO FILE UPLOADED"
            session['face_path'] = None
            return redirect(url_for('home'))

        image = Image.open(file).convert('RGB')
        image_np = np.array(image)

        faces = detector.detect_faces(image_np)
        if not faces:
            session['prediction'] = "NO FACE DETECTED!"
            session['face_path'] = None
            return redirect(url_for('home'))

        x, y, w, h = faces[0]['box']
        face = image_np[y:y+h, x:x+w]
        face_image = Image.fromarray(face).resize((128, 128))
        face_array = np.array(face_image) / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        prediction = model.predict(face_array)
        class_index = np.argmax(prediction)
        predicted_class = class_names[class_index]

        # Save cropped face image
        face_image.save("static/test_face.jpg")

        session['prediction'] = predicted_class
        session['face_path'] = "static/test_face.jpg"
        return redirect(url_for('home'))

    except Exception as e:
        session['prediction'] = f"Error: {str(e)}"
        session['face_path'] = None
        return redirect(url_for('home'))

def retrain_model():
    IMAGE_SIZE = (128, 128)
    EPOCHS = 10
    BATCH_SIZE = 8

    image_data = []
    labels = []
    class_names_local = sorted(os.listdir("image_classification_dataset"))
    detector = MTCNN()

    for idx, class_name in enumerate(class_names_local):
        class_dir = os.path.join("image_classification_dataset", class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path).convert("RGB")
                img_np = np.array(img)
                faces = detector.detect_faces(img_np)
                if not faces:
                    continue
                x, y, w, h = faces[0]['box']
                face = img_np[y:y+h, x:x+w]
                face_image = Image.fromarray(face).resize(IMAGE_SIZE)
                img_array = np.array(face_image) / 255.0
                image_data.append(img_array)
                labels.append(idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    if not image_data:
        return False, "No valid training data found."

    X = np.array(image_data)
    y = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model_new = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(class_names_local), activation='softmax')
    ])

    model_new.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_new.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE)

    model_new.save("person_classifier_model.h5")
    return True, f"Model retrained with {len(class_names_local)} classes."

@app.route('/retrain', methods=['POST'])
def retrain():
    success, message = retrain_model()
    if success:
        global model, class_names
        model = load_model("person_classifier_model.h5")
        class_names = sorted(os.listdir("image_classification_dataset"))
    session['status'] = message
    session['retrain_status'] = message  # New for HTML
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
