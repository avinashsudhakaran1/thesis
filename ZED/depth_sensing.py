########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample demonstrates how to capture a live 3D point cloud   
    with the ZED SDK and display the result in an OpenGL window.    
"""

import sys
import ogl_viewer.viewer as gl
import numpy as np
import pyzed.sl as sl
import math

if __name__ == "__main__":
    print("Running Depth Sensing sample ... Press 'Esc' to quit")

    init = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                 depth_mode=sl.DEPTH_MODE.ULTRA,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    init.set_from_svo_file('C:/Users/Fred/Desktop/Thesis/doorway.svo')
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
    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(len(sys.argv), sys.argv, camera_model, res)

    point_cloud = sl.Mat()
    normal_map = sl.Mat()
    depth = sl.Mat()
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


    nb_frames = zed.get_svo_number_of_frames()
    while viewer.is_available():
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            zed.get_position(zed_pose, sl.REFERENCE_FRAME.WORLD)
            zed.get_sensors_data(zed_sensors, sl.TIME_REFERENCE.IMAGE)
            zed_imu = zed_sensors.get_imu_data()
            viewer.updateData(point_cloud)
            
            # for row in point_cloud.get_data()[0]:
            #     freq_value = np.bincount(point_cloud).argmax()
            # zed.retrieve_measure(normal_map, sl.MEASURE_NORMALS)
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

                z_diff = abs(point_cloud_np[2] - tz)
                x_diff = point_cloud_np[0] - tx
                output_angle = (math.atan2(z_diff,x_diff) * 180/np.pi )  - 90
                print(f"Output: z_diff: {z_diff}, x_diff: {x_diff}, output angle {output_angle}\n")
                
                #output values of direction to file
                output_file.write(f"{svo_position},{z_diff},{x_diff},{output_angle}")   
                output_file.write("\n")   
            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break

    viewer.exit()
    zed.close()
