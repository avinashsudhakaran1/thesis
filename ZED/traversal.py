import sys
import ogl_viewer.viewer as gl
import numpy as np
import pyzed.sl as sl
import math
import numpy as np 
import matplotlib.pyplot as plt
import time
import random
import cv2

def main():
    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                 depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    init.set_from_svo_file('D:/ZED data/doorway.svo')
    zed = sl.Camera()
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    res = sl.Resolution()
    res.width = 1920
    res.height = 1080

    # Enable positional tracking with default parameters
    py_transform = sl.Transform()  # First create a Transform object for TrackingParameters object
    tracking_parameters = sl.PositionalTrackingParameters(_init_pos=py_transform)
    err = zed.enable_positional_tracking(tracking_parameters)

    camera_model = zed.get_camera_information().camera_model

    point_cloud = sl.Mat()
    depth = sl.Mat()
    view_image = sl.Mat()
    zed_pose = sl.Pose()
    zed_sensors = sl.SensorsData()

    f_in = open("traversal_input.csv","r")
    output_file = open("direction_output.csv","a")
    input_file = f_in.read()
    # convert file into dict mapping of frame to x,y tuple of goal state
    input_rows = input_file.split('\n')
    goal_dict = {}
    for row in input_rows:
        if row != '':
            val = row.split(',') #frame,x,y
            goal_dict[int(val[0])] = (int(val[1]) , int(val[2]))

    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second


    #### CREATE PLOT#######
    # to run GUI event loop
    # plt.ion()
    # plot_circle()


    #######################
    translation_left_to_center = zed.get_camera_information().calibration_parameters.T[0]
    nb_frames = zed.get_svo_number_of_frames()
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve the left image in sl.Mat
            zed.retrieve_image(view_image, sl.VIEW.LEFT)
            tracking_state = zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
            # zed.get_position(camera_pose, sl.REFERENCE_FRAME.CAMERA)
            zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
            zed_imu = zed_sensors.get_imu_data()
            # get the start time
            # start_time = time.time()
            cv2.imshow("Video",view_image.get_data())
            cv2.waitKey(1)
            
            # Get the distance between the center of the camera and the left eye
            # Retrieve and transform the pose data into a new frame located at the center of the camera
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                transform_pose(zed_pose.pose_data(sl.Transform()), translation_left_to_center)
            
            if(svo_position in goal_dict): #if current frame has a goal state allocated
                depth_distance = depth.get_data()[goal_dict[svo_position][1]][goal_dict[svo_position][0]]
                # print("DEPTH: " + str(depth_distance))
                point_cloud_np = point_cloud.get_data()[goal_dict[svo_position][1]][goal_dict[svo_position][0]]
                print(f"POINT CLOUD: {point_cloud_np}")
                # Display the translation and timestamp
                py_translation = sl.Translation()
                tx = round(zed_pose.get_translation(py_translation).get()[0], 3)
                ty = round(zed_pose.get_translation(py_translation).get()[1], 3)
                tz = round(zed_pose.get_translation(py_translation).get()[2], 3)
                print("Translation: Tx: {0}, Ty: {1}, Tz {2}, Timestamp: {3}".format(tx, ty, tz, zed_pose.timestamp.get_milliseconds()))

                py_orientation = sl.Orientation()
                ox = round(zed_pose.get_orientation(py_orientation).get()[0], 3) # get orientation of wheelchair
                print(ox)
                z_diff = abs(point_cloud_np[2] - tz)
                x_diff = point_cloud_np[0] - tx
                output_angle = (math.atan2(z_diff,x_diff) * 180/np.pi )  - 90 #-x (commented out for testing but may need to be added in for rotation of wheelchair)
                print(f"Output: z_diff: {z_diff}, x_diff: {x_diff}, output angle {output_angle}\n")
                
                if(not np.isnan(output_angle)):
                    update_plot(output_angle)


                #output values of direction to file
                output_file.write(f"{svo_position},{z_diff},{x_diff},{output_angle}")   
                output_file.write("\n")   
            # # get the end time
            # end_time = time.time()
            # # get the current fps
            # fps = 1 / (end_time - start_time)
            # # add current fps to total fps
            # total_fps += fps
            # # increment frame count
            # frame_count += 1
            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

    cv2.destroyAllWindows()
    zed.close()
    # calculate and print the average FPS
    # avg_fps = total_fps / frame_count
    # print(f"Average FPS: {avg_fps:.3f}") 

def transform_pose(pose, tx) :
    '''
    Translate the pose from the left eye of the camera to the center of the camera so turning gives correct
    orientation when rotation occurs
    '''
    transform = sl.Transform()
    transform.set_identity()
    # Translate the tracking frame by tx along the X axis
    transform.m[0][3] = tx
    # Pose(new reference frame) = M.inverse() * pose (camera frame) * M, where M is the transform between the two frames
    transform_inv = sl.Transform()
    transform_inv.init_matrix(transform)
    transform_inv.inverse()
    pose = transform_inv * pose * transform


height = 1000
width=1000

def update_plot(angle):
    #add 90 degrees to angle to reposition it to vertical and convert to rads
    angle_rad = (angle + 90)  * np.pi/180
    canvas = np.zeros((height,width,3), np.uint8)
    cv2.circle(canvas,((int(width/2)), (int(height/2))),int(height/2), color=(255,255,255), thickness=5)
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (450, 650)
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    cv2.putText(canvas, f"{angle:.2f}", org, font, fontScale, color, thickness, cv2.LINE_AA)

    line_start = (int(width/2), int(height/2)) #center of circle

    radius = int(width/2)
    if(angle >= 90):
        x = int(width/2) + int(abs(radius * np.cos(angle_rad)))
        y = int(height/2) - int(radius * np.sin(angle_rad)) 
    else:
        x = int(width/2) - int(radius * np.cos(angle_rad))
        y = int(radius * np.sin(angle_rad)) - int(height/2)

    

    line_end = (x, y)


    cv2.line(canvas,line_start,line_end,(255,255,255),5)


    cv2.imshow("Direction",canvas)
    cv2.waitKey(1)

if __name__ == "__main__":
    main()