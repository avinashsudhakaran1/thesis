import cv2
import os
import math
import numpy as np
import sys
import pyzed.sl as sl

def main():
    images = os.listdir('./export3')
    
    # Specify SVO path parameter
    init_params = sl.InitParameters()
    svo_input_path = r"C:/Users/Fred/Desktop/Thesis/doorway.svo"
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    # Prepare single image containers
    left_image = sl.Mat()
    depth_image = sl.Mat()

    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.FILL

    sys.stdout.write("Starting video....\n")

    nb_frames = zed.get_svo_number_of_frames()

    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)

            src_img = left_image.get_data()
            depth_img = depth_image.get_data()

            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
    #closing all open windows 
    cv2.destroyAllWindows()
    zed.close()
    return 0