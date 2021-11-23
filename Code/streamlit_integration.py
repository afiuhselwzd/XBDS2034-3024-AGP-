#libraries for streamlit
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import base64
from numpy.core.numeric import False_

#libraries for age/gender pred. algo.
import cv2 as cv
import math
import time

#function to open image files
@st.cache
def loadImg(imgFile):
    img = Image.open(imgFile)
    return img

#creating a selectbox to choose which video
def fileSelector(path='Vids'):
    file = os.listdir(path)
    selectFile = st.selectbox('Select a video', file)
    return os.path.join(path, selectFile)

#function used to download files
def download(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

#function for detection and bounding box
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

#function for live prediction
def predLive(cap):
    padding = 20
            
    # Read frame
    hasFrame, frame = cap.read()
    frameFace, bboxes = getFaceBox(faceNet, frame)

    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                    
        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        label = f'{gender}, {age}'
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
        frm_window.image(frameFace)

#function for pred. using img.
def predImg(cap):
    padding = 20
                
    # Read frame
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, cap)

    if not bboxes:
        st.write('No face found.')

    for bbox in bboxes:
        # print(bbox)
        face = cap[max(0,bbox[1]-padding):min(bbox[3]+padding,cap.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, cap.shape[1]-1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        st.write(f"Gender: {gender}, Confidence Score = {(genderPreds[0].max() * 100):.2f}" + '%')

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        st.write(f"Age: {age}, Confidence Score = {(agePreds[0].max() * 100):.2f}" + '%')
        st.write('')

        label = f'{gender}, {age}'
        cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    return frameFace, t

#loading pretrained libraries
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-15)', '(16-20)', '(21-24)', 
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

nav = st.sidebar.selectbox('Navigation', ['Home', 'Video Recordings', 'Age/Gender Prediction', 'About'])

if nav == 'Home':
    st.title('Online Age Prediction')
    st.subheader('What can you do?\n' + 'Using this website, you are able to:')
    st.markdown('\n1. View video recordings from the real-time age prediction application.\n'
              + '2. Use the age/gender prediction algorithm to make a prediction.\n'
              + '3. Learn more about the age/prediction code.')
    st.write('')
    imgHome = Image.open('sample-output.jpg')
    st.image(imgHome, width=750)

if nav == 'Video Recordings':
    st.title('Video Recordings')

    #explanation of page
    st.write("")
    st.markdown('You can view the recordings made by the real-time prediction model below. It will start recording when it detects someone entering the frame, and stops recording 5 seconds after someone exits the frame.')
    st.write("")

    #selectbox for available videos
    with st.container():
        filename = fileSelector()
        st.markdown(download(filename, 'Video'), unsafe_allow_html=True)

if nav == 'Age/Gender Prediction':
    st.title('Age/Gender Prediction')

    #explanation of page
    st.write('')
    st.markdown('By uploading a image below, you are able to get the predicted age and gender of people in it. Alternatively, you can also use a live webcam feed.')
    st.write('')
    
    cam = st.checkbox('Use live video feed')

    if cam:
        #creating "frm_window" to store video in array form, "cap" for accessing webcam
        frm_window = st.image([])
        cap = cv.VideoCapture(0)
        while cap:
            predLive(cap)
    else:
        #upload img
        st.subheader('Upload an Image')
        upFile = st.file_uploader('Upload an Image', type=['png', 'jpeg', 'jpg'])
        if upFile:
            fileType = {'File Name':upFile.name, 'File Type':upFile.type}
            st.write(fileType)
            imgFile = upFile
            img = loadImg(imgFile)
            st.image(img, width=750)
            
            #converting image into array and changing RGB -> BGR for improved prediction results
            cap = np.array(img)
            cv.imwrite('temp.jpg', cv.cvtColor(cap, cv.COLOR_RGB2BGR))
            cap = cv.imread('temp.jpg')

            #print image with bounding boxes + computation time
            frm, t = predImg(cap)
            st.image(frm)
            st.write("Computation Time : {:.3f}".format(time.time() - t) + 's')

if nav == 'About':
    st.title('About')

    #explanation about the model
    st.write("")
    st.subheader('Model')
    st.markdown('The code below uses pretrained libraries to predict the age and gender of a person. It will create a bounding box around a persons face and returns their predicted age range and gender.')
    st.write("")
    st.write("")
    
    #explanation about the video recordings
    st.subheader('Video Recordings')
    st.markdown('These video recordings are uploaded everytime there is a new recording, to allow for online viewing of videos without having access to the source file.')
    st.write("")
    st.write("")

    #explanation about the age prediction function
    st.subheader('Age/Gender Prediction')
    st.markdown('By uploading a photo, you are able to use the prediction algorithm to determine the predicted age and gender of people. Alternatively, you are also able to use a live webcam.')
    st.markdown('Further explanation of the code for the predictive algorithm can be found below.')
    st.write('')
    st.write("")

    #explanation of code
    st.subheader('Code')

    with st.container():
        st.write('')
        st.markdown('The libraries used for the real-time age prediction model')
        code = '''import cv2 as cv
import math
import time
import argparse
import datetime'''
        st.code(code, language='python')
        st.write("")

    with st.container():
        st.write('')
        st.markdown('The following code is used to predict the age and gender.')
        code = '''for bbox in bboxes:
    # print(bbox)
    face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
            max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

    blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
        
    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    # print("Age Output : {}".format(agePreds))
    print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

    label = "{},{}".format(gender, age)
    cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    cv.imshow("Age Gender Demo", frameFace)

    print("time : {:.3f}".format(time.time() - t))'''
        st.code(code, language='python')
        st.write("")

    with st.container():
        st.write('')
        st.markdown('The following code is used to detect and draw bounding boxes around a face.')
        code = '''def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes'''
        st.code(code, language='python')
        st.write("")

    with st.container():
        st.write('')
        st.markdown('The following code is used to load the pretrained models into the coding environment.')
        code = '''faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"'''
        st.code(code, language='python')
        url = 'https://github.com/ai-with-nur/Age-Gender-Prediction'
        st.write('Link to pretrained model files can be found [here](%s)' % url)
        st.write("")
        st.write("")

    # reference of code
    st.subheader('References')
    st.write('')
    st.write('https://github.com/ai-with-nur/Age-Gender-Prediction')
    st.write('https://docs.streamlit.io/library/api-reference/write-magic/st.write')
