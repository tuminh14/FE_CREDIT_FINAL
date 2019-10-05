import cv2
import math
import re
def rotation(image, angleInDegrees):
        h, w = image.shape[:2]
        img_c = (w / 2, h / 2)

        rot = cv2.getRotationMatrix2D(img_c, angleInDegrees, 1)

        rad = math.radians(angleInDegrees)
        sin = math.sin(rad)
        cos = math.cos(rad)
        b_w = int((h * abs(sin)) + (w * abs(cos)))
        b_h = int((h * abs(cos)) + (w * abs(sin)))

        rot[0, 2] += ((b_w / 2) - img_c[0])
        rot[1, 2] += ((b_h / 2) - img_c[1])

        outImg = cv2.warpAffine(image, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
        return outImg

def regex_VNtoEN(text):
  #to telex
  text=re.sub('[ÀÁÂÃĂẠẢẤẦẨẪẬẮẰẲẴẶăạảấầẩẫậắằẳẵặàáâã]','A',text)
  text=re.sub('[ÈÉÊèéêẸẺẼỀỀỂẹẻẽềềểỄỆễệ]','E',text)
  text=re.sub('[ìíĩỉịÌÍĨỈỊ]','I',text)
  text=re.sub('[òóõọỏôốồổỗộơớờởỡợÒÓÕỌỎÔỐỒỔỖỘƠỚỜỞỠỢ]','O',text)
  text=re.sub('[ùúũụủưứừửữựÙÚŨỤỦƯỨỪỬỮỰ]','U',text)
  text=re.sub('[ỳỵỷỹýỲỴỶỸÝ]','Y',text)
  text=re.sub('[Đđ]','D',text)
  return text.upper()

def regex_so(text):
  text= ''.join(re.findall('\d',text))
  return re.sub('\n','',text)

def regex_ten(text):
  text=regex_VNtoEN(text)
  text= ''.join(re.findall('\s*[A-Z,a-z]+\s*',text))
  return re.sub('\n','',text)

def regex_ngaySinh(text):
  text= ''.join(re.findall('[0-9,-]',text))
  return re.sub('\n','',text)

def regex_nguyenQuan(text):
  text=regex_VNtoEN(text)
  text= ''.join(re.findall('\s*[A-Z,a-z,0-9]+\s*',text))
  return re.sub('\n','',text)

def regex_noiDKHK(text):
  text=regex_VNtoEN(text)
  text= ''.join(re.findall('\s*[A-Z,a-z,0-9]+\s*',text))
  return re.sub('\n','',text)
