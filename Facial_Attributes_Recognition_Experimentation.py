import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adam

IMG_WIDTH = 178
IMG_HEIGHT = 218
IMG_WIDTH2 = 229
IMG_HEIGHT2 = 229

face_haar_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_alt.xml')

# Gender Identification
gender_model = load_model("./models/gender_identification_xception.h5", compile=False)
gender_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Face Mask Detection
fmd_model = load_model("./models/face_mask_xception.h5", compile=False)
fmd_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Glasses or no Glasses wearing
gw_model = load_model("./models/glasses_no_glasses_EfficientNetB7.h5", compile=False)
gw_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

gender = ("Man", "Woman")
mask = ("With Mask", "Without Mask")
glasses = ("With Glasses", "Without Glasses")

cam = cv2.VideoCapture(0)
while True:
    image = cam.read()[1]
    faces = face_haar_cascade.detectMultiScale(image, minNeighbors=5)
    for (x, y, w, h) in faces:
        face_image = image[y:y + h, x:x + w]
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 255), 1)
        # Resize for gender prediction
        face_image_gender = cv2.resize(face_image, (IMG_HEIGHT, IMG_WIDTH))
        # Resize for mask and glasses prediction
        face_image_mask_glasses = cv2.resize(face_image, (IMG_HEIGHT2, IMG_WIDTH2))

        # Preprocess for gender prediction
        face_image_gender = img_to_array(face_image_gender) / 255.0
        result_gender = gender_model.predict(np.expand_dims(face_image_gender, axis=0))
        gender_label = gender[np.argmax(result_gender)]
        gender_score = result_gender[0][np.argmax(result_gender)]

        # Preprocess for mask prediction
        face_image_mask_glasses = img_to_array(face_image_mask_glasses) / 255.0
        result_mask = fmd_model.predict(np.expand_dims(face_image_mask_glasses, axis=0))
        mask_label = mask[np.argmax(result_mask)]
        mask_score = result_mask[0][np.argmax(result_mask)]

        # Preprocess for glasses prediction
        result_glasses = gw_model.predict(np.expand_dims(face_image_mask_glasses, axis=0))
        glasses_label = glasses[np.argmax(result_glasses)]
        glasses_score = result_glasses[0][np.argmax(result_glasses)]

        # Display predictions
        cv2.putText(image, f"Gender: {gender_label} ({gender_score:.2f})", (x, y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(image, f"Mask: {mask_label} ({mask_score:.2f})", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(image, f"Glasses: {glasses_label} ({glasses_score:.2f})", (x, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    cv2.imshow("result", image)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()