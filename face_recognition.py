# Recognise Faces using some classification algorithm - like Logistic, KNN, SVM etc.
# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values we need to assign for each person
# 2. Read a video stream using opencv
# 3. extract faces out of it
# 4. use knn to find the prediction of face (int)
# 5. map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name
#Goal of Algorithm is to find out whose image is using given new image
import numpy as np
import cv2
import os
########## KNN CODE ############
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())
def knn(train, test, k=5):
	dist = []
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	#Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	#Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]
################################
#Initialize Camera
cap=cv2.VideoCapture(0)
#We are going to do face detection using haarcascade
#Face Detection
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
skip=0
dataset_path='./data/'
face_data=[]#face_data will be training data(X values of data),We will load all files
labels=[]#It will form Y values of data
class_id=0#Let class_id starts from zero,It is basically labels for the given file
names={}#To create Mapping between names and class_id
#Once we have processed one file ,we will increment our class_id by one,then 2 and so on
#Data Preparation
#Iterate over my directory
#In Command Prompt,type dir(It will show all the files of your directory ).For the same we use listdir in ubuntu.
for fx in os.listdir(dataset_path):#fx is some file,dataset_path is file path,os.listdir will search all files of data folder
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]#Take all the characters before dot
        print("Loaded  "+fx)
        data_item=np.load(dataset_path+fx)#There is a variable called data_item which loads this file
        face_data.append(data_item)
        #Create labels for the class
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)
        #We are going to concatenate all the items of the list into a single list
face_dataset=np.concatenate(face_data,axis=0)#X-train
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))#Y-train
print(face_labels)
print(type(face_labels))
print(face_dataset)
print(face_dataset.shape)
print(face_labels.shape)
#Our knn algorithm accepts one Training Matrix -So we should have X data and Y data combined in a matrix
Trainset=np.concatenate((face_dataset,face_labels),axis=1)#Means it will add 1 column to previous face_dataset
print(Trainset)
print(Trainset.shape)
#Testing
while True:
    ret,frame=cap.read()
    if ret==False:
        continue
    faces=face_cascade.detectMultiScale(frame,1.3,5)
    for face in faces:
        x,y,w,h=face
    #Get the face- Region Of Interest
        offset=10
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]#We have extracted our test frame 
        face_section=cv2.resize(face_section,(100,100))#We have resized it to a shape of 100 cross 100
    #Predicted Label(output)
        output=knn(Trainset,face_section.flatten())  
    #Display on the screen the name and rectangle around it
        Predicted_Name=names[int(output)]
        cv2.putText(frame,Predicted_Name,(x,y-17),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)#2 here is thickness,cv2.LINE_AA is for better look
       # Python: cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) 
   # org – Bottom-left corner of the text string in the image.
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Faces",frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
#Don't forget to transfer all .npy files from 6th folder to 7th folder. 