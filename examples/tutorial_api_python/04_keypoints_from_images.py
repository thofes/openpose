# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
from numpy.core.fromnumeric import shape
import json

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    try:
        # Windows Import
        if platform == "win32":
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/../../python/openpose/Release');
            os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
            import pyopenpose as op
        else:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append('../../python');
            # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
            # sys.path.append('/usr/local/python')
            from openpose import pyopenpose as op
    except ImportError as e:
        print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
        raise e

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="../../../examples/media/", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Add others in path?
    for i in range(0, len(args[1])):
        curr_item = args[1][i]
        if i != len(args[1])-1: next_item = args[1][i+1]
        else: next_item = "1"
        if "--" in curr_item and "--" in next_item:
            key = curr_item.replace('-','')
            if key not in params:  params[key] = "1"
        elif "--" in curr_item and "--" not in next_item:
            key = curr_item.replace('-','')
            if key not in params: params[key] = next_item

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir);
    
    print(imagePaths)


    

    start = time.time()
    list_keypoints = []
    keypoints_array = np.empty(0)
    # Process and display images
    for imagePath in imagePaths:
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))




        # wenn keine Pose im Bild etdeckt wird
        if datum.poseKeypoints is None: 
            keypoints = np.zeros(shape=(75))
            keypoints_numpy = np.zeros(shape=(75))
        # wenn mehrere Posen im Bild enteckt werden
        elif datum.poseKeypoints.shape != (1,25,3):
            keypoints_0 = datum.poseKeypoints[0,:,:]
            keypoints = np.reshape(keypoints_0, [75])
            keypoints_numpy = np.reshape(keypoints_0, [75])
        # Normalfall
        else:
            keypoints = np.reshape(datum.poseKeypoints, [75])
            keypoints_numpy = np.reshape(datum.poseKeypoints, [75])
        
        #für Verwendung mit Array
        keypoints_array = np.append(keypoints_array, keypoints_numpy)
        #für Verwendung mit List
        keypoint_list = keypoints.tolist()
        list_keypoints.append(keypoint_list)

        
        #print("Body keypoints: \n" + str(datum.poseKeypoints))

        if not args[0].no_display:
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            key = cv2.waitKey(15)
            if key == 27: break
    #text file oder json file        
    text_path = os.path.dirname(imagePath) + '/keypoints.txt'
    json_path = os.path.dirname(imagePath) + '/keypoints.json'

    """ Kommentieren, da aktuell nur ein Bild detektiert wird
    # check missing pics from YOLO and create empty list
    path_list = []
    if len(imagePaths) != 21:
        for i in imagePaths:
            if i[-6] is '/':
                x = i[-5]
            else:
                x = i[-6:-4]
            path_list.append(int(x))
    #fill the list for missing pics with empty arrays
        missing = [ele for ele in range(22) if ele not in path_list]
        missing_keypoints = np.zeros(shape=(75))
        keypoint_missing = missing_keypoints.tolist()
        missing.remove(0)
        
        for index in missing:
            list_keypoints.insert(index-1, keypoint_missing)
            keypoints_array = np.insert(keypoints_array, (index-1)*75, missing_keypoints)
    """

#Json File
    #with open(json_path, "w") as fp:
        #json.dump(list_keypoints, fp, indent=4)
    with open(os.path.dirname(imagePath) + '/keypoints.json', "w") as fp:
        json.dump(keypoints_array.tolist(), fp, indent=4)
    
#Text File
    line = str(list_keypoints)
    with open(text_path, 'a') as writefile:
        writefile.write(line)
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
