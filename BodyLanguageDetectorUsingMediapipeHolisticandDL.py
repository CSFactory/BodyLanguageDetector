#BODY_LANGUAGE_DETECTOR: FACE, RIGHT_HAND, LEFT_HAND, POSE_DETECTION
import cv2
import mediapipe as mp
import csv #to save data into csv files
import os #for folder structure #to access particular folder from import/export data
import numpy as np #for numerical calculations on arrays
import pandas as pd #pandas helps to deal with tabular data e.g: csv
from sklearn.model_selection import train_test_split #train_test_split helps to partition our coords.csv into training data and test(validation data)

#To train our classification model we need the following 4 dependencies.
#out of below 4, 1st 2 dependencies are required to create a machine learning pipeline
#
from sklearn.pipeline import make_pipeline #make_pipeline allows you to build an ML pipeline (There are different functions
# in pipeline to use on training data (features) and the test data)
from sklearn.preprocessing import StandardScaler #it standardizes the data (subtracts the mean and divide it by standard_deviation)
#standardization, ideally, get every data point on a level_basis. (One feature does not overshadow the other feature)
#Below 2 lines presents 4 ML classification models i.e. LogisticRegression, RidgeClassifier, RandomForestClassifier, GradientBoostingClassifier
#Test all the 4 ML classification models on our data and find the best one based on Test(Validation) Accuracy.
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pickle 

with open('body_language_ML_model.pkl', 'rb') as f: #Imported our RF ML trained model #rb: read binary
    model = pickle.load(f)

mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_holistic = mp.solutions.holistic # Mediapipe Solutions

cap = cv2.VideoCapture('1.mp4')
# Initiate holistic model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor Feed
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #Mediapipe processes RGB format, BGR->RGB conversion required
        image.flags.writeable = False        #This saves the image getting copied and helps us to render the same image.
        #When detection happened then we have to set this flag as TRUE. 
        
        # Make Detections
        results = holistic.process(image) #'results' variable contains all the processed features detected by Mediapipe-Holistic.
        # print(results.face_landmarks)
        
        # face_landmarks, pose_landmarks, left_hand_landmarks, right_hand_landmarks
        
        # Recolor image back to BGR for rendering
        image.flags.writeable = True   #Here detection happened so we have set this flag as TRUE. 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #Open-CV processes BGR format, RGB->BGR conversion required
        
        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 ) #DISPLAYING_FACE_LANDMARKS USING FACEMESH
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 ) #DISPLAYING_RIGHT_HAND_LANDMARKS

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 ) #DISPLAYING_LEFT_HAND_LANDMARKS

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )#DISPLAYING_POSE_LANDMARKS
                        
         #check how many landmarks, we need to handle that we have got from the Mediapipe-Holistic models i.e. FACE, RIGHT and LEFT HAND, POSE.
        #Set Up column names, so that you can refer each coordinate with a column name, while training your DL model for Overall Body
        #Language detection

        #First Column representing a CLASS which tells about different poses (like HAPPY, SAD and VICTORIOUS)
        # {Here you can add many number of extra poses as you want}, 
        # and rest of the columns are representing: COORDINATES (X,Y,Z,V(VISIBILITY))
        #Exception: FACE MODEL doesnot have a visibility coordinate whereby POSE_MODEL has a visibility coordinate.
        #So for FACE MODEL, we will have all zeroes in visibility coordinate.

        num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark) #results variable from BODYLANGUAGEDETECTORUSINGMEDIAPIPIEHOLISTICandDL.py
        #Total pose_landmarks = 33 and Total face_landmarks = 468
        #therefore, total number of coordinates (num_coords) = 468+33 = 501
        #So, we have to loop through all these number of coordinates.

        #each body language class (e.g.: HAPPY, SAD, VICTORIOUS) will have 501*4(x,y,z,v) = 2004 array elements.

        landmarks = ['class']
        for val in range(1, num_coords+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

        #Put these 2005 (1 column for class (Happy,sad,victorious) and 2004 columms for coordinates)
        #  landmarks into new file named coords.csv in a row-wise manner.
        #Run below three lines of code for the very first time to create a coords.csv.
       # with open('coords.csv', mode='w', newline='') as f: #creating coords.csv for the first time with class and with x1,y1,z1,v1 to x501,y501,z501,v501.
        #    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
         #   csv_writer.writerow(landmarks)

        #class_name = "WakandaForever" #Sad,Victorious,Drowsy,Angry,Irritated,Surprise,Fear etc. (Un-comment it when writing class data to coords.csv)
        #NOTE: Change this classname while recording for a particular class
        #We have to collect coordinates by making each face and pose for each class.
        #NOTE: there will be one row for one frame, if you record video for more than one frame for one class
        # (e.g.:HAPPY),then there will be number of rows for same class  = number of frames recorded or processed 
        # for that class coordinates recording.
         
          # Export coordinates
        try:
            # Extract Pose landmarks
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
            
            # Extract Face landmarks
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())
            
            # Concate rows
            row = pose_row+face_row
            
            #FROM line number: 125 to line number: 133 is for appending the data to coords.csv (Un-Comment it whenever new class 
            # data or similar class data needs to be inserted into coords.csv)
            # Append class name 
            #row.insert(0, class_name)
            
            # Export to CSV
            #Below three lines of code appends the row to the coords.csv for a particular class (e.g: HAPPY)
            #containing 2004 coordinates for both face and pose.
            # with open('coords.csv', mode='a', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(row) 

            ##NOTE: If we feel that the particular class (HAPPY,SAD,VICTORIOUS) doesnot perform well on the output
            # then, we need to performance tune by adding more frames data to the particular class of coords.csv 
            # for the particular class coordinates.
            # (MORE DATA = BETTER PERFORMANCE)

             # Make Detections on Real Time Data
                # 1. Detect Landmarks, 
                # 2. Predict Pose (aka class) based on landmarks coordinates
                # 3. Render Landmarks and Body Language Pose Using Open CV
            X = pd.DataFrame([row]) #take row data which contain coordinates of a frame x1,y1,z1,v1 to x501,y501,z501,v501
            body_language_class = model.predict(X)[0] #predict the class based on the input corrdinates of the frame
            #body_language_prob = model.predict_proba(X)[0] #probability of the predicted body language class based on input frame 
            #NOTE: predict_proba can help you to apply threshold if you want to
            #print(body_language_class, body_language_prob) 

             # Grab ear coords
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                        , [640,480]).astype(int))
            
            cv2.rectangle(image, 
                          (coords[0], coords[1]+5), 
                          (coords[0]+len(body_language_class)*20, coords[1]-30), 
                          (245, 117, 16), -1)
            cv2.putText(image, body_language_class, coords, 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
            
            # Display Class
            cv2.putText(image, 'CLASS'
                        , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, body_language_class.split(' ')[0]
                        , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # # Display Probability
            # cv2.putText(image, 'PROB'
            #             , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
            #             , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
        except:
            pass
        
        cv2.imshow('Raw Webcam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
