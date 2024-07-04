from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from tensorflow import keras
import io
from PIL import Image
import pillow_avif
import base64

HOME = os.getcwd()
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
print(HOME)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def custom_lambda(x):
    return x / 255

custom_objects = {'custom_lambda': custom_lambda}

def load():
    ### change file path if running locally
    model_216 = keras.models.load_model("./app/fire_segmentation.keras", custom_objects=custom_objects)

    inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
    s = tf.keras.layers.Lambda(custom_lambda)(inputs)

    #contraction
    c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = tf.keras.layers.Dropout(0.1)(c1)
    c1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)

    c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    c2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)

    c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    c3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)

    c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    c4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)

    c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = tf.keras.layers.Dropout(0.3)(c5)
    c5 = tf.keras.layers.Conv2D(256, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    #expansive
    u6 = tf.keras.layers.Conv2DTranspose(128, (2,2), strides=(2,2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)
    c6 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    
    u7 = tf.keras.layers.Conv2DTranspose(64, (2,2), strides=(2,2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)
    c7 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = tf.keras.layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)
    c8 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = tf.keras.layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = tf.keras.layers.Dropout(0.1)(c9)
    c9 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = tf.keras.layers.Conv2D(1, (1,1), activation='sigmoid')(c9)
    model_215 = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    for layer_216, layer_215 in zip(model_216.layers, model_215.layers):
        if layer_216.weights:
            layer_215.set_weights(layer_216.get_weights())

    model_215.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model_215

model = load()

@app.post("/")
async def predict(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(content={'error': 'No file part'}, status_code=400)

    ### change path if running locally
    dataset_path = "/app/dataset/test/demo"

    if not os.path.exists(dataset_path):
        print(f"The path {dataset_path} does not exist.")
    else:
        print(f"The path {dataset_path} exists.")

    file_path = os.path.join(dataset_path, file.filename)

    if not os.path.exists(file_path):
        print(f"The file path {file_path} does not exist.")
    else:
        print(f"The file path {file_path} exists.")

    img = imread(file_path)[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    img = img.astype(np.uint8) 

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img, axis=0)

    preds_test = model.predict(img_array, verbose=1)

    preds_test_t = (preds_test > 0.3).astype(np.uint8)

    predicted_mask = preds_test_t[0]

    if predicted_mask.shape[-1] == 1:
        predicted_mask = np.squeeze(predicted_mask, axis=-1)

    pred_img = Image.fromarray(predicted_mask * 255, mode='L')

    img_bytes = io.BytesIO()
    pred_img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    img_base64 = base64.b64encode(img_bytes.read()).decode('utf-8')

    mask = np.array(predicted_mask)

    true_count = np.count_nonzero(mask)

    ratio = true_count/(mask.size)

    if ratio > 0.15:
        category = 1
    elif ratio > 0.08:
        category = 2
    elif ratio > 0.05:
        category = 3
    elif ratio > 0:
        category = 4
    else:
        category = 5

    return {"ratio": ratio, "category": category, "img": img_base64}



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)