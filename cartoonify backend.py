# 1.2.2 BACK-END
from flask import Flask, render_template, request, json 
from werkzeug.utils import secure_filename
import requests
import cv2 #for image processing
import numpy as np #to store image
import sys
import matplotlib.pyplot as plt 
import os
import random
import numpy as np
from keras.models import model_from_json 
from keras.preprocessing import image
import h5py

# Open the .h5 file in read-only mode
f = h5py.File('C:/Users/akhil/OneDrive/Documents/Courses/SSDI/model (1).h5', 'r')

app = Flask(_____name____)

#gpath=''
#em1=''

def save(imagex, ImagePath):
    newName="cartoonified_Image_1"
    path1 = os.path.dirname(ImagePath)
    extension=os.path.splitext(ImagePath)[1]
    path = os.path.join(path1, newName+extension) 
    #gpath=path
    cv2.imwrite(path, imagex)
    em1 = emotion(path)
    # print(em1)
    return path, em1


# // 4

def cartoonify(ImagePath, filter):
    originalmage = cv2.imread(ImagePath)
    originalmage = cv2.cvtColor(originalmage, cv2.COLOR_BGR2RGB) 
    w,h,o=originalmage.shape

    if originalmage is None:
        print("Can not find any image. Choose appropriate file") 
        sys.exit()

    Resized1 = cv2.resize(originalmage, (h,w))

    grayScaleImage= cv2.cvtColor(originalmage, cv2.COLOR_BGR2GRAY) 
    ReSized2 = cv2.resize(grayScaleImage, (h,w))

    smoothGrayScale = cv2.medianBlur(grayScaleImage, 5)
    ReSized3= cv2.resize(smoothGrayScale, (h,w))

    getEdge = cv2.adaptiveThreshold(smoothGrayScale, 255, cv2.ADAPTIVE_THRESH_MEAN_C, CV2.THRESH_BINARY, 9, 9)
    
    ReSized4 = cv2.resize(getEdge, (h,w))

    colorImage = cv2.bilateralFilter(originalmage, 9, 300, 300)
    ReSized5 = cv2.resize(colorImage, (h,w))

    cartoonImage = cv2.bitwise_and(colorImage, colorImage, mask=getEdge)

    ReSized6= cv2.resize(cartoonImage, (h,w))
    images=[Resized1, ReSized2, ReSized3, ReSized4, ReSized5, ReSized6]
    fig, axes = plt.subplots (1,2, figsize=(8,8), subplot_kw={'xticks': [], 'yticks':[]}, gridspec_kw = dict(hspace=0.1, wspace=0.1))

    for i, ax in enumerate(axes.flat):
        if(i==0):
            ax.imshow(images[i], cmap='gray')
        else:
            ax.imshow(images [filter], cmap='gray')
    
    name, em1 = save(images [filter], ImagePath)
    return name, em1

# // 5
def emotion(ImagePath):
    model = model_from_json(open('model.json', "r").read())
    #load weights
    model.load_weights('model.h5')
    face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    test_img=cv2.imread(ImagePath)
    gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    faces_detected = face_haar_cascade.detectMultiScale (gray_img, 1.32, 5) 
    img_pixels=[0]

    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), thickness=7)
        roi_gray-gray_img[y:y+w,x:x+h]
        roi_gray=cv2.resize(roi_gray, (48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims (img_pixels, axis = 0) 
        img_pixels /= 255

        # print(img_pixels) 
    if(img_pixels.any()==0):
        return "Not Found" 
    a=random.randint(0,2)
    predictions = model.predict(img_pixels)
    max_index = np.argmax (predictions[0])
    angry=('You look angry and frustrated. Please calm down', 'Angry 2')
    disgust=('cap1', 'cap2')
    fear=('cap1', 'cap2')
    happy=('cap1', 'cap2')
    sad=('cap1', 'cap2')
    surprise = ('cap1', 'cap2')
    neutral=('cap1', 'cap2')

    # emotions= (angry, disgust, fear, happy, sad, surprise, neutral) 
    emotions = ('You look angry and frustrated. Please calm down', 'A look of disgust. Seems like you need to go to your happy place.', 'You look frightened. Everything is going to be okay',
'You look so happy. Keep going have a good day.',
'You look sad. Cheer up and watch a movie',
'Looks like someone has got a big surprise',
'You seem pretty calm without any emotions right now.') 
    predicted_emotion = emotions[max_index]
    # predicted_emotion=predict[a]
    return predicted_emotion

# // 6
# def c_emotion(ImagePath):
# model = model_from_json (open('model.json',"r").read())
# #load weights
# model.load_weights('model.h5')
# face_haar_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') test_img=cv2.imread(ImagePath)
# gray_img= cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# faces_detected = face_haar_cascade.detectMultiScale(gray_img,1.32, 5) img_pixels=[0]
# for (x,y,w,h) in faces_detected:
# cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,0,0), thickness=7)
# roi_gray-gray_img[y:y+w,x:x+h]
# roi_gray=cv2.resize(roi_gray, (48,48))
# img_pixels = image.img_to_array(roi_gray)
# img_pixels = np.expand_dims (img_pixels, axis = 0) img_pixels / 255
# # print(img_pixels) if(img_pixels.any()==0):
# return "Not Found" a=random.randint(0,2)
# predictions = model.predict(img_pixels)
# max_index = np.argmax (predictions[0])
# # emotions= (angry, disgust, fear, happy, sad, surprise, neutral) emotions ('<p id="emtn">You look angry and frustrated. Please calm down<br> <br>Quite an angry look. Peace out<br>The HULK look</p>',
# '<p id="emtn">A look of disgust. Seems like you need to go to your happy place.<br> <br>My face when you make bad jokes <br> Ewwwwwwwww! </p>',
# '<p id="emtn">You look frightened. Everything is going to be okay<br><br>The time I saw Conjuring alone<br>Fear of growing up</p>',
# '<p id="emtn">You look so happy. Keep going have a good day.<br><br>when the weekend arrives<br>Just smile away your problems</p>',
# '<p id="emtn">You look sad. Cheer up and watch a movie<br><br>Frown like a clown<br> Tears are words that need to be written</p>',
# '<p id="emtn">Looks like someone has got a big surprise<br><br>When I see an empty box on the street<br>The Michael Scott look</p>',
# '<p id="emtn">You seem pretty calm without any emotions right now.<br><br>Calm before the storm<br>Peace out like BUDDHA</p>')
# predicted_emotion = emotions [max_index]
# # predicted_emotion=predict[a]
# return predicted_emotion

# // 7
@app.route("/", methods = ["GET", "POST"])
def new():
    if request.method == "POST":
        f = request.files['fileupload']
        f.save(secure_filename (f.filename))
        filter=int(request.values.get("filter")) 
        ImagePath="./"+str(f.filename)
        nam, em1=cartoonify (ImagePath, filter) 
        em=emotion(ImagePath)
        #em1-emotion (gpath)
        di=dict()
        di[0]=nam
        di[1]=em 
        di[2]=em1 
        di[3]=ImagePath 
        print(di) 
        return di
    return render_template("index.html")

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT' 
    return response

if __name__ == "__main__":
    app.run(debug=True)
