#!flask/bin/python
import base64
from flask import Flask, jsonify,request
import numpy as np

import cv2
import os
import time
# construct the argument parse and parse the arguments
confthres = 0.3
nmsthres = 0.1




#object detection function
def get_labels(labels_path):
    # load the COCO class labels our YOLO model was trained on
    lpath=os.path.sep.join([yolo_path, labels_path])

    print(yolo_path)
    LABELS = open(lpath).read().strip().split("\n")
    return LABELS


def get_weights(weights_path):
    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([yolo_path, weights_path])
    return weightsPath

def get_config(config_path):
    configPath = os.path.sep.join([yolo_path, config_path])
    return configPath

def load_model(configpath,weightspath):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configpath, weightspath)
    return net

def do_prediction(image,net,LABELS):
    
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    #print(layerOutputs)
    end = time.time()

    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            # print(scores)
            classID = np.argmax(scores)
            # print(classID)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confthres:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])

                confidences.append(float(confidence))
                classIDs.append(classID)

    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confthres,
                            nmsthres)

    # TODO Prepare the output as required to the assignment specification
    # ensure at least one detection exists
    result=[]
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            print("detected item:{}, accuracy:{}, X:{}, Y:{}, width:{}, height:{}".format(LABELS[classIDs[i]],
                                                                                             confidences[i],
                                                                                             boxes[i][0],
                                                                                             boxes[i][1],
                                                                                             boxes[i][2],
                                                                                             boxes[i][3]))
            #print(LABELS[classIDs[i]])
            #print(confidences[i])
            #print(boxes[i][0])
            #print(boxes[i][1])
            #print(boxes[i][2])
            #print(boxes[i][3])
            result.append([LABELS[classIDs[i]],confidences[i],boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]])
            
    return result
    
    


#web server
app = Flask(__name__, static_url_path = "")
#auth = HTTPBasicAuth()

@app.route("/")#test
def hello():
    return "object detection"


@app.route('/api/object_detection',methods=["POST"])

def create_task():
    all_data=str(request.data)
    
    data_list=all_data.split("\\\\\"")
    del all_data #release RAM
   
    image=data_list[3]
    
    id=data_list[7]
    del data_list#release RAM
   
    image=base64.b64decode(image)#decode
    
    #don't do IO
    #with open("a.jpg", 'wb') as f:
        #f.write(image1)
        #f.close()
    
    #img = cv2.imread("a.jpg")

    #to image
    image = np.fromstring(image, np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image=np.array(image) 
    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # load the neural net.  Should be local to this method as its multi-threaded endpoint
    nets = load_model(CFG, Weights)
    pre_json=do_prediction(image, nets, Lables)#predict label
    del nets,image#release RAM
    object=[]
    if len(pre_json)!=0:
        a={}
        for k in range(len(pre_json)):
            
            
            a={
            "label":pre_json[k][0],
            "accuracy":pre_json[k][1],
            "rectangle":{
            "height":pre_json[k][5],
            "left":pre_json[k][2],#x
            "top":pre_json[k][3],#y
            "width":pre_json[k][4]}}
            #a=json.dumps(a, sort_keys=True, indent=4, separators=(',', ': '))

            object.append(a)
    
    if len(pre_json)==0:
        a={}
        a={
        "label":"",
        "accuracy":0,
        "rectangle":{
        "height":0,
        "left":0,#x
        "top":0,#y
        "width":0}
        }
        #a=json.dumps(a, sort_keys=True, indent=4, separators=(',', ': '))
        object.append(a)
    
    del pre_json#release RAM
    output={
        "id":id,
        "object":object
    } 
    #output==json.dumps(output, sort_keys=True, indent=4, separators=(',', ': '))
    del id#release RAM
    del object#release RAM
    return  jsonify(output), 201

if __name__ == '__main__':
    
    yolo_path  = "yolo_tiny_configs/"
    ## Yolov3-tiny versrion
    labelsPath= "coco.names"
    cfgpath= "yolov3-tiny.cfg"
    wpath= "yolov3-tiny.weights"

    Lables=get_labels(labelsPath)
    CFG=get_config(cfgpath)
    Weights=get_weights(wpath)






    app.run(host="0.0.0.0",
        port=1024,
            debug = True)

