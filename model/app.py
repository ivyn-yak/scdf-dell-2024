from skimage.io import imread
from skimage.transform import resize
import os
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
import io
from PIL import Image
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

model = keras.models.load_model("./fire_segmentation.keras", custom_objects=custom_objects)

@app.post("/")
async def predict(file: UploadFile = File(...)):
    if not file:
        return JSONResponse(content={'error': 'No file part'}, status_code=400)

    file_path = os.path.join("../dataset/test/demo", file.filename)

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
    else:
        category = 4

    return {"ratio": ratio, "category": category, "img": img_base64}



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)