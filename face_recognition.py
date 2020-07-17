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
face_data=[]
labels=[]
class_id=0
names={}


#Data Preparation
#Iterate over my directory
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id]=fx[:-4]
        print("Loaded  "+fx)
        data_item=np.load(dataset_path+fx)
        face_data.append(data_item)
        #Create labels for the class
        target=class_id*np.ones((data_item.shape[0],))
        class_id+=1
        labels.append(target)
face_dataset=np.concatenate(face_data,axis=0)#X-train
face_labels=np.concatenate(labels,axis=0).reshape((-1,1))#Y-train
print(face_labels)
print(type(face_labels))
print(face_dataset)
print(face_dataset.shape)
print(face_labels.shape)
Trainset=np.concatenate((face_dataset,face_labels),axis=1)
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
        face_section=frame[y-offset:y+h+offset,x-offset:x+w+offset]
        face_section=cv2.resize(face_section,(100,100))
        output=knn(Trainset,face_section.flatten())  
        Predicted_Name=names[int(output)]
        cv2.putText(frame,Predicted_Name,(x,y-17),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.imshow("Faces",frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
