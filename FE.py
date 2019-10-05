import cv2
import numpy as np
from google.cloud import vision
from google.cloud.vision import types
import pandas as pd
import json
from helpFunction import *
import face_recognition as fr
import os
import re
import matplotlib.pyplot as plt

class FaceRecognition:
    __image=None
    __image_selfie=None
    __gray_cmnd=None
    __gray_self=None
    __face_cascade=None
    image_clone=None
    accuracy=0

    def __init__(self,image_cmnd,image_selfie ):
        self.__image=image_cmnd
        self.__image_selfie=image_selfie
        self.image_clone=self.__image
        try:
            self.__face_cascade = cv2.CascadeClassifier('pre_train_model/haarcascade_frontalface_default.xml')
            self.__eye_cascade = cv2.CascadeClassifier('pre_train_model/haarcascade_eye.xml')
        except:
            assert self.__face_cascade==None , 'fail to load file pre_train_model/haarcascade_frontalface_default.xml '
            assert self.__eye_cascade==None , 'fail to load file pre_train_model/haarcascade_eye.xml '

        dimention= True

        self.__image=rotation(self.__image, 90)
        self.__width_image=self.__image.shape[1]
        self.__height_image=self.__image.shape[0]
    

        try:
            faces = self.__face_cascade.detectMultiScale(cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY), 1.5, 5)
            # faces = fr.face_locations(self.__image,  number_of_times_to_upsample=2, model="cnn")
            #     faces = fr.face_location(image,2,'cnn')
            x_cmnd,y_cmnd,w_cmnd,h_cmnd = faces[0]
            face_cmnd =image[y_cmnd:y_cmnd+h_cmnd, x_cmnd:x_cmnd+w_cmnd]
            face_encoded = fr.face_encodings(face_cmnd, num_jitters=100)[0]
    
            if x_cmnd > int(0.55195*self.__height_image):
                dimention=False
            
        except:
            self.__image=rotation(self.__image, -180)
            self.__width_image=self.__image.shape[1]
            self.__height_image=self.__image.shape[0]

            try:
                faces = self.__face_cascade.detectMultiScale(cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY), 1.5, 5)
                # faces = fr.face_locations(self.__image,  number_of_times_to_upsample=2, model="cnn")
            #     faces = fr.face_location(image,2,'cnn')
                x_cmnd,y_cmnd,w_cmnd,h_cmnd = faces[0]
                face_cmnd =image[y_cmnd:y_cmnd+h_cmnd, x_cmnd:x_cmnd+w_cmnd]
                # face_encoded = fr.face_encodings(face_cmnd, num_jitters=100)[0]
                
                if x_cmnd > int(0.55195*self.__height_image):
                    dimention=False
                    
            except:
                pass

        #crop
        if dimention:
            h_top=int(0.227*self.__height_image)
            h_bot=int(0.227*self.__height_image)
            w_left=int(0.38996*self.__height_image)
            w_right=int(0.55195*self.__height_image)

            self.__image=self.__image[h_top:self.__height_image-h_bot,w_left:self.__width_image-w_right]
            self.__width_image=self.__image.shape[1]
            self.__height_image=self.__image.shape[0]
        else:
            h_top=int(0.25*self.__height_image)
            h_bot=int(0.2*self.__height_image)
            w_right=int(0.38996*self.__height_image)
            w_left=int(0.53195*self.__height_image)

            self.__image=self.__image[h_top:self.__height_image-h_bot,w_left:self.__width_image-w_right]
            self.__width_image=self.__image.shape[1]
            self.__height_image=self.__image.shape[0]

        try:
            self.__face_cascade = cv2.CascadeClassifier('pre_train_model/haarcascade_frontalface_default.xml')

        except:
            assert self.__face_cascade==None , 'fail to load file pre_train_model/haarcascade_frontalface_default.xml '

        self.image_clone=self.__image

        self.__gray_cmnd = cv2.cvtColor(self.__image, cv2.COLOR_BGR2GRAY)
        self.__gray_self = cv2.cvtColor(self.__image_selfie, cv2.COLOR_BGR2GRAY)

    def __face_distance_to_conf(self,face_distance, face_match_threshold=0.4):
        if face_distance > face_match_threshold:
            range = (1.0 - face_match_threshold)
            linear_val = (1.0 - face_distance) / (range * 2.0)
            return linear_val
        else:
            range = face_match_threshold
            linear_val = 1.0 - (face_distance / (range * 2.0))
            return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))

    def compareFace(self):
        #detect face 
        try:
            faces_cmnd = self.__face_cascade.detectMultiScale(self.__gray_cmnd, 1.5, 5)
            faces_self = self.__face_cascade.detectMultiScale(self.__gray_self, 1.4, 5)
            # faces_cmnd = fr.face_locations(self.__gray_cmnd,  number_of_times_to_upsample=5, model="hog")
            # faces_self = fr.face_locations(self.__gray_self,  number_of_times_to_upsample=2, model="hog")
            try:
                x_cmnd,y_cmnd,w_cmnd,h_cmnd = faces_cmnd[0]
                face_cmnd = self.__image[y_cmnd:y_cmnd+h_cmnd, x_cmnd:x_cmnd+w_cmnd]
                face_cmnd=cv2.cvtColor(face_cmnd,cv2.COLOR_BGR2RGB)
            except:
                print('We do not find out any face in your identification! Please try again')
                return False

            try:
                x_self,y_self,w_self,h_self = faces_self[0]
                face_self = self.__image_selfie[y_self:y_self+h_self, x_self:x_self+w_self]
                face_self=cv2.cvtColor(face_self,cv2.COLOR_BGR2RGB)
            except:
                print('We do not find out any face in your selfie! Please try again')
                return False
        except:
            print('Somethings went wrong! Please try again later')
            return False

        try:
            if (face_cmnd.shape == face_self.shape):
                difference = cv2.subtract(face_cmnd, face_self)
                b, g, r = cv2.split(difference)
                if cv2.countNonZero(b) == 0 and cv2.countNonZero(g) == 0 and cv2.countNonZero(r) == 0:
                    print('Somethings wrong here, two faces you give are exactly the same! Try again!')
                else:
                    cmnd_encoded = fr.face_encodings(face_cmnd,num_jitters=100)[0]
                    self_encoded = fr.face_encodings(face_self,num_jitters=100)[0]
                    match = fr.compare_faces([cmnd_encoded],self_encoded)[0]
                    distance = fr.face_distance([cmnd_encoded],self_encoded)[0]
                    if (distance >= 0.1):
                        print('Dont do that!!')
            else:
                w = min(face_cmnd.shape[0], face_self.shape[0])
                h = min(face_cmnd.shape[1], face_self.shape[1])
                dim = (w,h)
                face_cmnd_resize = cv2.resize(face_cmnd,dim)
                face_self_resize = cv2.resize(face_self,dim)
            #     face_cmnd_resize = ((np.log(face_cmnd_resize+1)/(np.log(1+np.max(face_cmnd_resize))))*255).astype('uint8')
            #     face_self_resize = ((np.log(face_self_resize+1)/(np.log(1+np.max(face_self_resize))))*255).astype('uint8')
                cmnd_encoded = fr.face_encodings(face_cmnd_resize,num_jitters=100)[0]
                self_encoded = fr.face_encodings(face_self_resize,num_jitters=100)[0]
                match = fr.compare_faces([cmnd_encoded],self_encoded)[0]
                distance = fr.face_distance([cmnd_encoded],self_encoded)[0]

                self.accuracy=self.__face_distance_to_conf(distance)
                return True
        except Exception as e:
            print(e)
            return False


class OCR:
    __image = None
    __width_image=0
    __height_image=0
    __client=None
    text_API_OCR_so=''
    text_API_OCR_ten0=''
    text_API_OCR_ten1=''
    text_API_OCR_ngaySinh=''
    text_API_OCR_nguyenQuan0=''
    text_API_OCR_nguyenQuan1=''
    text_API_OCR_noiDKHK0=''
    text_API_OCR_noiDKHK1=''
    ls_so0=['02'
    ,'01'
    ,'20'
    ,'280'
    ,'27'
    ,'22'
    ,'03'
    ,'30'
    ,'20'
    ,'27'
    ,'24'
    ,'36'
    ,'26'
    ,'25'
    ,'19'
    ,'37'
    ,'12'
    ,'10'
    ,'17'
    ,'18'
    ,'14'
    ,'230'
    ,'285'
    ,'14'
    ,'21'
    ,'31'
    ,'15'
    ,'12'
    ,'11'
    ,'35'
    ,'13'
    ,'29'
    ,'090'
    ,'06'
    ,'16'
    ,'21'
    ,'32'
    ,'245'
    ,'38'
    ,'33'
    ,'16'
    ,'13'
    ,'26'
    ,'22'
    ,'16'
    ,'18'
    ,'34'
    ,'36'
    ,'23'
    ,'19'
    ,'19'
    ,'33'
    ,'36'
    ,'05'
    ,'38'
    ,'15'
    ,'07'
    ,'04'
    ,'04'
    ,'08'
    ,'07'
    ,'095'
    ,'08']
    ls_so1=['02'
    ,'01'
    ,'20'
    ,'280'
    ,'27'
    ,'22'
    ,'03'
    ,'30'
    ,'20'
    ,'27'
    ,'24'
    ,'36'
    ,'26'
    ,'25'
    ,'19'
    ,'37'
    ,'12'
    ,'10'
    ,'17'
    ,'18'
    ,'14'
    ,'231'
    ,'285'
    ,'14'
    ,'21'
    ,'31'
    ,'15'
    ,'12'
    ,'11'
    ,'35'
    ,'13'
    ,'29'
    ,'091'
    ,'06'
    ,'16'
    ,'21'
    ,'32'
    ,'245'
    ,'38'
    ,'33'
    ,'16'
    ,'13'
    ,'26'
    ,'22'
    ,'16'
    ,'18'
    ,'34'
    ,'36'
    ,'23'
    ,'19'
    ,'19'
    ,'33'
    ,'36'
    ,'05'
    ,'38'
    ,'15'
    ,'07'
    ,'04'
    ,'04'
    ,'08'
    ,'07'
    ,'095'
    ,'08']
    ls=['HO CHI MINH',
    'HA NOI',
    'DA NANG',
    'BINH DUONG',
    'DONG NAI',
    'KHANH HOA',
    'HAI PHONG',
    'LONG AN',
    'QUANG NAM',
    'BA RIA VUNG TAU',
    'DAK LAK',
    'CAN THO',
    'BINH THUAN  ',
    'LAM DONG',
    'THUA THIEN HUE',
    'KIEN GIANG',
    'BAC NINH',
    'QUANG NINH',
    'THANH HOA',
    'NGHE AN',
    'HAI DUONG',
    'GIA LAI',
    'BINH PHUOC',
    'HUNG YEN',
    'BINH DINH',
    'TIEN GIANG',
    'THAI BINH',
    'BAC GIANG',
    'HOA BINH',
    'AN GIANG',
    'VINH PHUC',
    'TAY NINH',
    'THAI NGUYEN',
    'LAO CAI',
    'NAM DINH',
    'QUANG NGAI',
    'BEN TRE',
    'DAK NONG',
    'CA MAU',
    'VINH LONG',
    'NINH BINH',
    'PHU THO',
    'NINH THUAN',
    'PHU YEN',
    'HA NAM',
    'HA TINH',
    'DONG THAP',
    'SOC TRANG',
    'KON TUM',
    'QUANG BINH',
    'QUANG TRI',
    'TRA VINH',
    'HAU GIANG',
    'SON LA',
    'BAC LIEU',
    'YEN BAI',
    'TUYEN QUANG',
    'DIEN BIEN',
    'LAI CHAU',
    'LANG SON',
    'HA GIANG',
    'BAC KAN',
    'CAO BANG']
    __face_cascade=None
    __eye_cascade=None
    image_result=None
    status=[]
    __check_list = []

    def __init__(self,image):
        self.__image=image.copy()
        self.__width_image=image.shape[1]
        self.__height_image=image.shape[0]

        

    def __detectText(self,img):    
        image = vision.types.Image(content=cv2.imencode('.jpg', img)[1].tostring())
        response = self.__client.text_detection(image=image)
        texts = response.text_annotations
        
        df = pd.DataFrame(columns=['locale','description'])
        for text in texts:
            df = df.append(
                dict(
                    locale=text.locale,
                    description=text.description
                ),
                ignore_index=True
            )
        return df['description'][0]

    def detect_ocr(self):
        #delete green
        self.image_result=self.__image.copy()
        image=self.__image.copy()
        width_image=self.__width_image
        height_image=self.__height_image
        for h in range(height_image):
            for w in range(width_image):
                (b,g,r)=image[h][w].astype('int16')
                if ((g-b)>15) and ((g-r)>15):
                    image[h][w]=(255,255,255)
        
        #crop to OCR
        #so cmnd
        h_so=int(height_image*0.11538)
        w_so=int(height_image*0.646)
        x_so=int(height_image*0.753846)
        y_so=int(height_image*0.2153846)
        API_image_so=image[y_so:y_so+h_so,x_so:x_so+w_so]
        #ten cm
        h_ten=int(height_image*0.1385)
        w_ten=int(height_image*0.8)
        x_ten=int(height_image*0.6615)
        y_ten=int(height_image*0.323)
        API_image_ten0=image[y_ten:y_ten+h_ten,x_ten:x_ten+w_ten]
        #ten cm
        h_ten1=int(height_image*0.1077)
        w_ten1=int(height_image*1.03077)
        x_ten1=int(height_image*0.4923)
        y_ten1=int(height_image*0.43077)
        API_image_ten1=image[y_ten1:y_ten1+h_ten1,x_ten1:x_ten1+w_ten1]

        h_ngaySinh=int(height_image*0.1077)
        w_ngaySinh=int(height_image*0.5)
        x_ngaySinh=int(height_image*0.8385)
        y_ngaySinh=int(height_image*0.5230769)
        API_image_ngaySinh=image[y_ngaySinh:y_ngaySinh+h_ngaySinh,x_ngaySinh:x_ngaySinh+w_ngaySinh]

        h_nguyenQuan=int(height_image*0.1077)
        w_nguyenQuan=int(height_image*0.7385)
        x_nguyenQuan=int(height_image*0.815385)
        y_nguyenQuan=int(height_image*0.61538)
        API_image_nguyenQuan0=image[y_nguyenQuan:y_nguyenQuan+h_nguyenQuan,x_nguyenQuan:x_nguyenQuan+w_nguyenQuan]

        h_nguyenQuan1=int(height_image*0.1077)
        w_nguyenQuan1=int(height_image*1.03077)
        x_nguyenQuan1=int(height_image*0.4923)
        y_nguyenQuan1=int(height_image*0.7077)
        API_image_nguyenQuan1=image[y_nguyenQuan1:y_nguyenQuan1+h_nguyenQuan1,x_nguyenQuan1:x_nguyenQuan1+w_nguyenQuan1]

        h_noiDKHK=int(height_image*0.1077)
        w_noiDKHK=int(height_image*0.60385)
        x_noiDKHK=int(height_image*0.95)
        y_noiDKHK=int(height_image*0.784615)
        API_image_noiDKHK0=image[y_noiDKHK:y_noiDKHK+h_noiDKHK,x_noiDKHK:x_noiDKHK+w_noiDKHK]

        h_noiDKHK1=int(height_image*0.1077)
        w_noiDKHK1=int(height_image*1.1577)
        x_noiDKHK1=int(height_image*0.4023)
        y_noiDKHK1=int(height_image*0.877)
        API_image_noiDKHK1=image[y_noiDKHK1:y_noiDKHK1+h_noiDKHK1,x_noiDKHK1:x_noiDKHK1+w_noiDKHK1]


        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'VisionAPIDemo/ServiceAccountToken.json'
        self.__client = vision.ImageAnnotatorClient()

        try:
            self.text_API_OCR_so=regex_so(self.__detectText(API_image_so))
        except:
            pass

        try:
            self.text_API_OCR_ten0=regex_ten(self.__detectText(API_image_ten0))
        except:
            pass

        try:
            self.text_API_OCR_ten1=regex_ten(self.__detectText(API_image_ten1))
        except:
            pass

        try:
            self.text_API_OCR_ngaySinh=regex_ngaySinh(self.__detectText(API_image_ngaySinh))
        except:
            pass

        try:
            self.text_API_OCR_noiDKHK0=regex_noiDKHK(self.__detectText(API_image_noiDKHK0))
        except:
            pass

        try:
            self.text_API_OCR_nguyenQuan0=regex_nguyenQuan(self.__detectText(API_image_nguyenQuan0))
        except:
            pass  

        try:
            self.text_API_OCR_nguyenQuan1=regex_nguyenQuan(self.__detectText(API_image_nguyenQuan1))
        except:
            pass  

        try:
            self.text_API_OCR_noiDKHK1=regex_noiDKHK(self.__detectText(API_image_noiDKHK1))
        except:
            pass  

        self.__check_list.append(self.text_API_OCR_so)
        self.__check_list.append(self.text_API_OCR_ten0)
        self.__check_list.append(self.text_API_OCR_ten1)
        self.__check_list.append(self.text_API_OCR_ngaySinh)
        self.__check_list.append(self.text_API_OCR_nguyenQuan0)
        self.__check_list.append(self.text_API_OCR_nguyenQuan1)
        self.__check_list.append(self.text_API_OCR_noiDKHK0)
        self.__check_list.append(self.text_API_OCR_noiDKHK1)
    
    def check_valid_day(self):
        check=False
        if re.search('^(3[01]|[12][0-9]|0?[1-9])-(1[0-2]|0?[1-9])-(?:[0-9]{2})?[0-9]{2}$', self.text_API_OCR_ngaySinh):
            split_text = self.text_API_OCR_ngaySinh
            split_text = split_text.split('-')
            check_text = split_text[-1]
            if (2019 - int(check_text,10) >= 18):
                check= True
            else:
                h_ngaySinh=int(self.__height_image*0.1077)
                w_ngaySinh=int(self.__height_image*0.5)
                x_ngaySinh=int(self.__height_image*0.8385)
                y_ngaySinh=int(self.__height_image*0.5230769)
                cv2.rectangle(self.image_result,(x_ngaySinh,y_ngaySinh),(x_ngaySinh+w_ngaySinh,y_ngaySinh+h_ngaySinh),(255,0,0),2)
                
            self.status.append(check)
        return check

    def check_valid_hometown(self):
        check = False
        for x in self.ls:
            split_text = self.__check_list[-4] + " " + self.__check_list[-3]
            split_text = split_text.split(' ')
            check_text = split_text[-3] + " " + split_text[-2] + " " + split_text[-1]
            if x in check_text:
                check= True
                break
            else:
                pass
        if check == False:
            h_nguyenQuan=int(self.__height_image*0.1077)
            w_nguyenQuan=int(self.__height_image*0.7385)
            x_nguyenQuan=int(self.__height_image*0.815385)
            y_nguyenQuan=int(self.__height_image*0.61538)
            h_nguyenQuan1=int(self.__height_image*0.1077)
            w_nguyenQuan1=int(self.__height_image*1.03077)
            x_nguyenQuan1=int(self.__height_image*0.4923)
            y_nguyenQuan1=int(self.__height_image*0.7077)
            cv2.rectangle(self.image_result,(x_nguyenQuan,y_nguyenQuan),(x_nguyenQuan+w_nguyenQuan,y_nguyenQuan+h_nguyenQuan),(0,255,0),2)
            cv2.rectangle(self.image_result,(x_nguyenQuan1,y_nguyenQuan1),(x_nguyenQuan1+w_nguyenQuan1,y_nguyenQuan1+h_nguyenQuan1),(0,255,0),2)
        
        self.status.append(check)
        return check

    def check_valid_address(self):
        check = False
        for x in self.ls:
            split_text = self.__check_list[-2] + " " + self.__check_list[-1]
            split_text = split_text.split(' ')
            check_text = split_text[-3] + " " + split_text[-2] + " " + split_text[-1]
            if x in check_text:
                check= True
                break
            else:
                pass

        if check == False:
            h_noiDKHK=int(self.__height_image*0.1077)
            x_noiDKHK=int(self.__height_image*0.95)
            y_noiDKHK=int(self.__height_image*0.784615)
            h_noiDKHK1=int(self.__height_image*0.1077)
            w_noiDKHK=int(self.__height_image*0.60385)
            w_noiDKHK1=int(self.__height_image*1.1577)
            x_noiDKHK1=int(self.__height_image*0.4023)
            y_noiDKHK1=int(self.__height_image*0.877)

            cv2.rectangle(self.image_result,(x_noiDKHK,y_noiDKHK),(x_noiDKHK+w_noiDKHK,y_noiDKHK+h_noiDKHK),(0,0,255),2)
            cv2.rectangle(self.image_result,(x_noiDKHK1,y_noiDKHK1),(x_noiDKHK1+w_noiDKHK1,y_noiDKHK1+h_noiDKHK1),(0,0,255),2)
        self.status.append(check)

        return check
    
    def check_valid_idNumber(self):
        index=-1
        for i,x in enumerate(self.ls):
            if x in (self.text_API_OCR_noiDKHK0+self.text_API_OCR_noiDKHK1):
                index=i
                break
        if index==-1:
            h_so=int(self.__height_image*0.11538)
            w_so=int(self.__height_image*0.646)
            x_so=int(self.__height_image*0.753846)
            y_so=int(self.__height_image*0.2153846)
            cv2.rectangle(self.image_result,(x_so,y_so),(x_so+w_so,y_so+h_so),(0,0,255),2)
            self.status.append(False)
            return False
        else :
            if self.ls_so0[index]==self.text_API_OCR_so[:len(self.ls_so0[index])] or self.ls_so1[index]==self.text_API_OCR_so[:len(self.ls_so1[index])]:
                self.status.append(True)
                return True
            else:
                h_so=int(self.__height_image*0.11538)
                w_so=int(self.__height_image*0.646)
                x_so=int(self.__height_image*0.753846)
                y_so=int(self.__height_image*0.2153846)
                cv2.rectangle(self.image_result,(x_so,y_so),(x_so+w_so,y_so+h_so),(0,0,255),2)
                self.status.append(False)
                return False
                