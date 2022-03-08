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
    # Process and display images
    for imagePath in imagePaths:
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))
        print("PAth: ", imagePath)
        print("Daetnytp: ", type(datum.poseKeypoints))


        # fehlt noch ein Teil, der prüt ob nur eine Pose gefunden wird

        # wenn keine Pose im Bild etdeckt wird
        if datum.poseKeypoints is None: 
            print("BP2")
            keypoints = np.zeros(shape=(75))
            print(keypoints)
        # wenn mehrere Posen im Bild enteckt werden
        elif datum.poseKeypoints.shape != (1,25,3):
            keypoints_0 = datum.poseKeypoints[0,:,:]
            keypoints = np.reshape(keypoints_0, [75])
            print("aha: ", keypoints.shape )
        # Normalfall
        else:
            print("Shape: ", datum.poseKeypoints.shape)
            keypoints = np.reshape(datum.poseKeypoints, [75])
        print("Shape_new: ", keypoints.shape)
        keypoint_list = keypoints.tolist()
        print("list: ", keypoint_list)
        list_keypoints.append(keypoint_list)

        #line = str(keypoint_list)
        #print("Line: ", line)
        
        print("Body keypoints: \n" + str(datum.poseKeypoints))
        #with open('/content.gdrive/MyDrive/Fotos/OP/Test/test.txt', 'a') as writefile:
            #writefile.write(line)
            #writefile.write(('%g ' * len(line)).rstrip() % line  +'\n')

        if not args[0].no_display:
            cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
            key = cv2.waitKey(15)
            if key == 27: break
    text_path = os.path.dirname(imagePath) + '/keypoints.txt'
    print("Path: " , text_path)
    print("list: ", list_keypoints)

    
    # check missing pics from YOLO and create empty list
    path_list = []
    if len(imagePaths) != 21:
        print("Bild fehlt")
        for i in imagePaths:
            if i[-6] is '/':
                x = i[-5]
            else:
                x = i[-6:-4]
            path_list.append(int(x))
    #fill the list for missing pics with empty arrays
        print("Path_list: ", path_list)
        missing = [ele for ele in range(22) if ele not in path_list]
        missing_keypoints = np.zeros(shape=(75))
        keypoint_missing = missing_keypoints.tolist()
        missing.remove(0)
        print("missing :", missing)
        
        for index in missing:
            print("index: ", index-1)
            print("keypoints: ", keypoint_missing)
            list_keypoints.insert(index-1, keypoint_missing)
        print("list: ", list_keypoints)

    line = str(list_keypoints)
    with open(text_path, 'a') as writefile:
        writefile.write(line)
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
