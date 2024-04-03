import cv2
import numpy as np
import os
# coding=utf-8
import urllib
import urllib.request
import hashlib

#加载训练数据集文件
recogizer=cv2.face.LBPHFaceRecognizer_create()
recogizer.read('trainer/trainerCat.yml')
names=[]
warningtime = 0

from PIL import Image, ImageDraw, ImageFont
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "STSONG.TTF", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

#准备识别的图片
def face_detect_demo(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度
    # 这里要写绝对路径
    face_detector=cv2.CascadeClassifier('C:/Users/33718/Desktop/face/catface/data/haarcascades/haarcascade_frontalcatface_extended.xml')
    # face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(100,100),(300,300))
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,)
    #face=face_detector.detectMultiScale(gray)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=2)
        cv2.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(0,255,0),thickness=1)
        # 人脸识别
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        #print('标签id:',ids,'置信评分：', confidence)
        if confidence > 80:
            global warningtime
            warningtime += 1
            if warningtime > 100:
               # warning()
               warningtime = 0
            cv2.putText(img, 'unkonw', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
        else:
            img = cv2ImgAddText(img, str(names[ids-1]), x + 10, y - 10, (255, 0, 0), 30)
            # cv2.putText(img,str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 1)
    cv2.imshow('result',img)
    #print('bug:',ids)

def name():
    path = './data/photos/'
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[1])
       names.append(name)


name()

# 摄像头检测
# cap=cv2.VideoCapture(0)
# cap = cv2.VideoCapture('1.mp4')
# while True:
#     flag,frame=cap.read()
#     if not flag:
#         break
#     face_detect_demo(frame)
#     if ord(' ') == cv2.waitKey(10):
#         break


frame = cv2.imread('1.jpg')
while True:
    # 调用人脸检测函数
    face_detect_demo(frame)

    # 等待按键或者一段时间后继续下一次循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#  作者：晓宜
#  🌈🌈🌈
#  个人简介：互联网大厂Java准入职，阿里云专家博主，csdn后端优质创作者，算法爱好者
#  🌙🌙🌙
#  小红书|csdn|牛客 同名 
#  希望可以帮助到大家
#  ❤️❤️❤️
#  你的关注是我前进的动力😊
