import pyzed.sl as sl
import math
import numpy as np
import sys
import cv2
import time
import PIL.Image as Image

def main():
    #################   CAMERA   ###########################
    zed = sl.Camera()

    #################   INITIAL PARAMETERS   #####################
    init_params = sl.InitParameters()
    init_params.set_from_svo_file('C:/Users/Fred/Desktop/Thesis/indoor.svo')
    init_params.camera_fps = 30
    init_params.sdk_verbose = True # Enable the verbose mode
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Use ULTRA depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD1080

    ############## OPEN CAMERA #################
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()


    ########################   RUNTIME PARAMS   ############################
    runtime_params = sl.RuntimeParameters(confidence_threshold=50, texture_confidence_threshold=100)

    ##############   TRANSFORMS  ###############
    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m
    
    res = sl.Resolution()
    res.width = 720
    res.height = 404


    camera_info = zed.get_camera_information()
    ##############  ITEMS TO DISPLAY   ######################
    image = sl.Mat() # rgb image
    depth_measure = sl.Mat(camera_info.camera_resolution.width, camera_info.camera_resolution.height, sl.MAT_TYPE.F32_C1)
    depth_display = sl.Mat()



    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        # Retrieve left image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # Retrieve depth map. Depth is aligned on the left image
        zed.retrieve_measure(depth_measure, sl.MEASURE.DEPTH)
        depth_ocv = depth_measure.get_data()
        # Print the depth value at the center of the image
        # print(depth_ocv[int(len(depth_ocv)/2)][int(len(depth_ocv[0])/2)])
        
        zed.retrieve_image(depth_display, sl.VIEW.DEPTH)
        # Use get_data() to get the numpy array
        image_depth_array = depth_display.get_data()
        # image_depth_array = image_depth_array[:,:,3]
        print(image_depth_array)
        # image_depth = Image.fromarray(image_depth_array)

        # # Naming a window
        # cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)
        
        # # Using resizeWindow()
        # cv2.resizeWindow("Resized_Window", 720, 404)
        # Display the depth view from the numpy array
        cv2.imshow("Resized_Window", image_depth_array)
        

        # depth_image_rgba = depth.get_data()
        # depth_image = cv2.cvtColor(depth_image_rgba, cv2.COLOR_RGBA2RGB)
        # cv2.imshow('depth',depth_image)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
        
        # Get and print distance value in mm at the center of the image
        # We measure the distance camera - object using Euclidean distance
        # x = round(image.get_width() / 2)
        # y = round(image.get_height() / 2)
        # err, point_cloud_value = point_cloud.get_value(x, y)

        # distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
        #                      point_cloud_value[1] * point_cloud_value[1] +
        #                      point_cloud_value[2] * point_cloud_value[2])

        # point_cloud_np = point_cloud.get_data()
        # point_cloud_np.dot(tr_np)

        # if not np.isnan(distance) and not np.isinf(distance):
        #     print("Distance to Camera at ({}, {}) (image center): {:1.3} m".format(x, y, distance), end="\r")
        #     # Increment the loop
        #     i = i + 1
        # else:
        #     print("Can't estimate distance at this position.")
        #     print("Your camera is probably too close to the scene, please move it backwards.\n")
        svo_position = zed.get_svo_position()
        print(svo_position)
        sys.stdout.flush()
    if zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
        print("SVO end has been reached. Looping back to first frame")
        zed.set_svo_position(0)
        cv2.destroyAllWindows()
        

    ####### CLOSE CAMERA #############
    zed.close()

if __name__ == "__main__":
    main()