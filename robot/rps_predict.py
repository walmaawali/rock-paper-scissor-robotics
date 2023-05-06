import cv2
import time
import numpy as np
from pymycobot.mycobot import MyCobot

# Please change this variable to the name of your CNN model
model_name = "rps_model.pb"

# Load the CNN model and print the layers
cnn = cv2.dnn.readNetFromTensorflow('models/'+model_name)
print("CNN model was successfully read. Model layers: \n", cnn.getLayerNames())

# Connect to a camera (please attach a USb camera with the robot)
cam = cv2.VideoCapture(0)

# Connect with myCobot
robot = MyCobot('/dev/ttyAMA0', 1000000)
if robot.is_controller_connected() == 1:
    print("Robot was successfully connected")
# Define poses
home_pose = [0,0,0,0,0,0]
rock_pose = [0,-60,146,-78,35,-18]
paper_pose = [0,0,90,0,40,-37]
scissor_pose=[90,0,0,0,35,-34]
poses = [paper_pose, rock_pose, scissor_pose, home_pose]
speed = 15

def scissor_gripper():
    for i in range(3):
        robot.set_gripper_state(1, 30)
        time.sleep(1)
        robot.set_gripper_state(0,30)
        time.sleep(1)
    robot.set_gripper_state(0,30)
    
robot.send_angles(poses[-1], speed)
# Declare the labels of RPS
rps_labels = ['Paper', "Rock", "Scissor"]

while True:
    _, img = cam.read()
    
    # perfrom pre-processing of the acquired image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32)

    input_blob = cv2.dnn.blobFromImage(
      image=gray,
      scalefactor=1.0/255,
      size=(150, 150),  # img target size
      crop=True  # center crop
      )
      
    cnn.setInput(input_blob)
    
    # Predict the class of the image
    out = cnn.forward()
    print("CNN prediction: \n")
    print("* shape: ", out.shape)
    # get the predicted class ID
    rps_class_id = np.argmax(out)
    label = rps_labels[rps_class_id]

    # get confidence
    confidence = out[0][rps_class_id]
    print("* class ID: {}, label: {}".format(rps_class_id, label))
    print("* confidence: {:.4f}\n".format(confidence))
    print("===========================================================")
    img = cv2.putText(img, label, (20,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0),2,cv2.LINE_AA)
    img = cv2.putText(img, str(confidence), (20,40), cv2.FONT_HERSHEY_PLAIN, 1, \
                      (255,0,0),2,cv2.LINE_AA)
    cv2.imshow('Video',img)
    
    k = cv2.waitKey(1)
    if k == ord('s'):
        cv2.imwrite('images/img'+str(int(time.time()))+'.jpg', img)
        robot.send_angles(poses[rps_class_id], speed)
        time.sleep(3)
        if rps_class_id == 2:
            scissor_gripper()
        robot.send_angles(poses[-1], speed)
    elif k != -1:
        break
    

    
cam.release()
cv2.destroyAllWindows()
