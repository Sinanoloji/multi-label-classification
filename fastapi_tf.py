import uvicorn
from fastapi import FastAPI,File,UploadFile
import tensorflow as tf
import numpy as np


app = FastAPI()

#if model is "*.keras", you must use "predict.tolist()[0]".
#if model is "*.h5", you muse use "predict[0]". 
model = tf.keras.models.load_model("model.h5")

def image_decode(image):
    img = tf.image.decode_image(image, channels=3)
    img = tf.image.resize(img,size= [224,224])
    img = img/255.0
    img = tf.expand_dims(img,axis=0)
    return img

def classification(data):

    color_pre = np.argmax(data[0])
    color = ['Black', 'Blue', 'Brown', 'Green', 'Grey', 'Mixed',
              'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    clr_dict = {i: color[i] for i in range(len(color))}

    season_pre = np.argmax(data[1])
    season = ['Fall', 'Spring', 'Summer', 'Winter']
    ssn_dict = {i: season[i] for i in range(len(season))}

    usage_pre = np.argmax(data[2])
    usage = ['Casual', 'Ethnic', 'Formal', 'Party', 'Smart Casual', 'Sports', 'Travel']
    usg_dict = {i: usage[i] for i in range(len(usage))}

    sub_pre = np.argmax(data[3])
    sub_cat = ['Bottomwear', 'Dress', 'Topwear']
    sub_dict = {i: sub_cat[i] for i in range(len(sub_cat))}

    puan = (np.max(data[0])+np.max(data[1])+np.max(data[2])+np.max(data[3])) / 4
    accuracy = int(100*puan)

    return [clr_dict[color_pre],ssn_dict[season_pre],sub_dict[sub_pre], usg_dict[usage_pre],str(accuracy)]

@app.post("/predict")
async def server(file: UploadFile = File(...)):
    image = await file.read()
    image = image_decode(image)
    predict = model.predict(image)
    prediction = classification(predict)

    return {"accuracy":prediction[4],"color": prediction[0],"season":prediction[1],"subcategory":prediction[2],"usage":prediction[3]}


if __name__ == "__main__":
    uvicorn.run(app,port=8000)