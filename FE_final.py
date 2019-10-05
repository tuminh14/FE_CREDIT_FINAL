import cv2 
import FE
import pyrebase
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os, io
from google.cloud import vision
import json
from firebase import firebase
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import db
import time
config = {
    "apiKey": "AIzaSyDlACLl0vF4Lei5Pgj1-gARAe--i6RgCM8",
    "authDomain": "codeschoolfecredit.firebaseapp.com",
    "databaseURL": "https://codeschoolfecredit.firebaseio.com",
    "projectId": "codeschoolfecredit",
    "storageBucket": "codeschoolfecredit.appspot.com",
    "messagingSenderId": "63196416274",
    "appId": "1:63196416274:web:dd45311b509d773f5f4bcb"
}

firebase = pyrebase.initialize_app(config)

db = firebase.database()



cmnd=cv2.imread('image/CMND/tung_cmnd.jpg')
selfie=cv2.imread('image/Selfie/tung_shelphie.jpg')

FR=FE.FaceRecognition(cmnd,selfie)
FR.compareFace()
# OCR_detect=FE.OCR(image)

# OCR_detect.detect_ocr()
# OCR_detect.check_valid_day()
# OCR_detect.check_valid_hometown()
# OCR_detect.check_valid_address()

# print(OCR_detect.status)

# cv2.imwrite('result.jpg',OCR_detect.image_result)


def stream_handler(message):
    # print(message["event"]) # put
    # print(message["path"]) # /-K7yGTTEp7O549EzTYtI
    # print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}
    ref = db.child("InfoImage").get()
    if(ref.val()[-1]['Status'] == 'NoCheck'):
        # print(message["event"]) # put
        # print(message["path"]) # /-K7yGTTEp7O549EzTYtI
        # print(message["data"]) # {'title': 'Pyrebase', "body": "etc..."}
        # ref = db.child("InfoImage").get()
        nameUri = ref.val()[-1]['NameUri']
        number = len(ref.val()) - 1
        print(nameUri)
        time.sleep(60)
        storage = firebase.storage()
        storage.child("ImageCMND/"+nameUri).download("image/CMND/" + nameUri)
        storage.child("ImageSelfie/"+nameUri).download("image/Selfie/" + nameUri)
        print("Downloaded")
        cmnd = cv2.imread('image/CMND/'+ nameUri)
        selfie = cv2.imread('image/Selfie/'+ nameUri)
        while(True):
            FR=FE.FaceRecognition(cmnd,selfie)
            FR.compareFace()
            if(FR.accuracy >= 0.5):
                cv2.imwrite('image/Out_image/' + nameUri, cmnd)
            else:
                cv2.imwrite('image/Out_image/' + nameUri, cmnd)
            print("Writed")
            #update
            if(FR.accuracy >= 0.5):
                db.child("InfoImage").child(number).update({"Status": "Check"}) #update status 
                db.child("InfoImage").child(number).update({"allow": "true"}) #update allowed
            else:
                db.child("InfoImage").child(number).update({"Status": "Check"}) #update status 
                db.child("InfoImage").child(number).update({"allow": "False"}) #update allowed
                break
            
            OCR_detect=FE.OCR(image)
            OCR_detect.detect_ocr()
            OCR_detect.check_valid_idNumber()
            OCR_detect.check_valid_day()
            OCR_detect.check_valid_hometown()
            OCR_detect.check_valid_address()
            if(False in OCR_detect.status):
                db.child("InfoImage").child(number).update({"Status": "Check"}) #update status 
                db.child("InfoImage").child(number).update({"allow": "False"}) #update allowed
                break
            else:
                db.child("InfoImage").child(number).update({"Status": "Check"}) #update status 
                db.child("InfoImage").child(number).update({"allow": "False"}) #update allowed
                break
        cv2.imwrite('image/Out_image/' + nameUri, OCR_detect.image_result)

        #push image to database
        storage = firebase.storage()
        storage.child("Images/" + nameUri).put("image/Out_image/" + nameUri)
        print("Push success!!!")



  
        
# Realtime streaming
my_stream = db.child("InfoImage").stream(stream_handler)


