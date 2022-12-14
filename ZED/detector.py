import cv2
import os
import math
import numpy as np
import sys
import pyzed.sl as sl
import time

img_array = []
def main():
    # images = os.listdir('./export3')
    
    # Specify SVO path parameter
    svo_input_path = r"D:/ZED data/doorway.svo"
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.METER  # Use milliliter units (for depth measurements)
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

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
    point_cloud = sl.Mat()

    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.FILL

    sys.stdout.write("Starting video....\n")

    nb_frames = zed.get_svo_number_of_frames()

    output_file = open("door_location.csv","w")
    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second

    #OUTPUT AS VIDEO
    video = cv2.VideoWriter('detection.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 10, (1920, 1080))
    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()
            print(svo_position)
            # get the start time
            start_time = time.time()
            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            zed.retrieve_image(depth_image, sl.VIEW.DEPTH)
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
            src_img = left_image.get_data()
            depth_img = depth_image.get_data()

            line_arr = houghlines(src_img)
            valid_line_arr = find_gaps(line_arr,depth_img,point_cloud,svo_position, output_file)
            if(len(valid_line_arr) > 0):
                img_out = drawlines(src_img, valid_line_arr,svo_position)
                video.write(img_out[:,:,:3])
            
            # get the end time
            end_time = time.time()
            # get the current fps
            fps = 1 / (end_time - start_time)
            # add current fps to total fps
            total_fps += fps
            # increment frame count
            frame_count += 1
            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
   
    # for image in img_array:
    #     image = image[:,:,:3]
    #     video.write(image)
    
    #closing all relevant out
    video.release()
    output_file.close()
    cv2.destroyAllWindows()
    zed.close()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}") 
    return 0
    


def drawlines(img, arr,pos):
    for line in arr:
        pt1_l = line[0]
        pt2_l = line[1]

        cv2.line(img, pt1_l, pt2_l, (0,0,255), 5, cv2.LINE_AA)
    
    cv2.imshow("Image with lines", img)
    # path = f"D:/ZED data/zed_doorwayboundary/{pos}.png"
    # cv2.imwrite(path,img)
    cv2.waitKey(1)
    return img

valid_line_arr = []
def find_gaps(line_arr,depth_img, point_cloud,svo_position, output_file):
    prev = None

    door_error_thresh = 20
    doorway_width_thresh = 550
    open_distance_threshold = 0.5 #(metres)
    
    #sort line array based on the x value of the first tuple (start point x) so that we can scan from left to right
    line_arr.sort(key=lambda tup: tup[0])

    
    if(len(line_arr) > 1): #can only perform gap finding if at least two lines found
        for i,line in enumerate(line_arr):
            if(i > 0): #skip the first line
                pt_1 = line[0] #start of line
                pt_2 = line[1] #end of line

                prev_pt_1 = prev[0] #start of prev line
                prev_pt_2 = prev[1] #end of prev line

                if(abs(pt_1[0] - prev_pt_1[0]) < doorway_width_thresh and abs(pt_1[0] - prev_pt_1[0]) > 200): #if the difference in x values of the two lines are legit
                    #check the depth values here
                    if ( int((pt_1[0] + pt_2[0])/2) + door_error_thresh <= depth_img.shape[1]): #if the threshold doesnt push it past left edge of the image
                        midpoint_cur = ( int((pt_1[0] + pt_2[0])/2) + door_error_thresh , int((pt_1[1] + pt_2[1])/2) )   #calculate midpoint of right line and move point to the right a bit more
                    else:
                        midpoint_cur = ( int((pt_1[0] + pt_2[0])/2) , int((pt_1[1] + pt_2[1])/2) )
                    
                    if ( int((pt_1[0] + pt_2[0])/2) + door_error_thresh >= 0): #if the threshold doesnt push it past right edge of the image
                        midpoint_prev = ( int((prev_pt_1[0] + prev_pt_2[0])/2) - door_error_thresh , int((prev_pt_1[1] + prev_pt_2[1])/2) )   #calculate midpoint of left line and move point to the left a bit more
                    else:
                        midpoint_prev = ( int((prev_pt_1[0] + prev_pt_2[0])/2), int((prev_pt_1[1] + prev_pt_2[1])/2) )   #calculate midpoint of left line and move point to the left a bit more

                    mid_midpoint = int((midpoint_cur[0] + midpoint_prev[0]) / 2)

                    err, left_pc_val = point_cloud.get_value(midpoint_prev[0],midpoint_prev[1])
                    left_distance = math.sqrt(left_pc_val[0] * left_pc_val[0] + left_pc_val[1] * left_pc_val[1] + left_pc_val[2] * left_pc_val[2])
                    err, right_pc_val = point_cloud.get_value(midpoint_cur[0],midpoint_cur[1])
                    right_distance = math.sqrt(right_pc_val[0] * right_pc_val[0] + right_pc_val[1] * right_pc_val[1] + right_pc_val[2] * right_pc_val[2])

                    err, middle_pc_val = point_cloud.get_value(mid_midpoint,midpoint_cur[1]) #mid mid point and using y value of current (could also be prev, redundant)
                    middle_distance = math.sqrt(middle_pc_val[0] * middle_pc_val[0] + middle_pc_val[1] * middle_pc_val[1] + middle_pc_val[2] * middle_pc_val[2])

                    #filter depth img into single channel
                    # filtered_depth_img = depth_img[:,:,1]
                    # #find the horizontal line of pixels at the y position of the midpoint of the vert lines
                    # depth_arr = filtered_depth_img[midpoint_prev[1]]
                    # mid_depth_check = depth_arr[mid_midpoint]
                    # left_depth_check = depth_arr[midpoint_prev[0]]
                    # right_depth_check = depth_arr[midpoint_cur[0]]
                    # avg_front_plane = int((left_depth_check + right_depth_check) / 2)
                    #filter arr into just the width of the door
                    # depth_arr = depth_arr[midpoint_prev[0] : midpoint_cur[0]]

                    # if(avg_front_plane - mid_depth_check  > open_door_thresh):
                    

                    if(middle_distance - min(left_distance,right_distance)  > open_distance_threshold):
                        #clear old state
                        valid_line_arr.clear()
                        # if(left_distance <= 2):
                        # print(f"LEFT {left_distance}")
                        # print(f"RIGHT {right_distance}")
                        #we have a valid gap so append the lines we want to keep on the image
                        valid_line_arr.append(prev)
                        valid_line_arr.append(line)
                        # Get and print distance value in m at the center of the image
                        # We measure the distance camera - object using Euclidean distance
                        # print(midpoint_cur[0])
                        # print(midpoint_cur[1])
                        err, pc_val = point_cloud.get_value(midpoint_cur[0],midpoint_cur[1])

                        distance = math.sqrt(pc_val[0] * pc_val[0] + pc_val[1] * pc_val[1] + pc_val[2] * pc_val[2])
                        # print(distance)   

                        #output coordinates of door to file
                        output_text = str(svo_position) + "," + str(mid_midpoint) + "," + str(midpoint_cur[1])
                        output_file.write(output_text)   
                        output_file.write("\n")              


            prev = line
    # valid_line_arr = line_arr
    return valid_line_arr


#https://www.delftstack.com/howto/python/opencv-line-detection/#:~:text=To%20detect%20the%20lines%20present,function%20is%20the%20given%20image.
def houghlines(src_img):
    # cv2.imshow('Original Image',src_img)

    dst_img = cv2.Canny(src_img, 20, 250, None, 3)

    lines = cv2.HoughLines(dst_img, 1, np.pi / 180, 150, None, 0, 0)
    line_arr = []
    try:
        for i in range(0, len(lines)):
            rho_l = lines[i][0][0]
            theta_l = lines[i][0][1]
            a_l = math.cos(theta_l)
            b_l = math.sin(theta_l)
            x0_l = a_l * rho_l
            y0_l = b_l * rho_l
            pt1_l = (int(x0_l + 1000*(-b_l)), int(y0_l + 1000*(a_l)))
            pt2_l = (int(x0_l - 1000*(-b_l)), int(y0_l - 1000*(a_l)))


            ##### TODO: ADD CODE FOR OTHER SCENARIOS (x value neg, x or y value greater than image width or height)
            if(pt1_l[1] < 0): #if y value is negative
                gradient = (pt2_l[1] - pt1_l[1]) / (pt2_l[0] - pt1_l[0])  #y2-y1 / x2-x1 to get slope
                c = pt2_l[1] - gradient * pt2_l[0]
                pt1_l = (int(-c/gradient), 0)

            #delta x should always be less than delta y to be a sort of vertical line
            delta_X = abs(pt2_l[0] - pt1_l[0]) 
            delta_Y = abs(pt2_l[1] - pt1_l[1])
            angle_deg = (math.atan2(delta_X, delta_Y) * 180 / math.pi) #angle from vertical axis
            if(angle_deg < 20):
                line_arr.append((pt1_l,pt2_l))
    except TypeError: #doesnt always detect lines so can return Nonetype
        pass

    return line_arr

def line_length(p1,p2):
    return math.sqrt(math.pow((p1[0]-p2[0]),2) + math.pow((p1[1]-p2[1]),2))
    

#loops through array and finds the length of the largest portion of the array with the same depth value
def count_longest_line(arr):
    count = 0 #running count of points
    largest_count = 0 #largest num of points stored
    
    prev_val = arr[0] #start at position 0 of array
    for cur_value in arr: #loop through each array position
        if prev_val == cur_value:
            count += 1
        else: #if not same value, reset count by first checking whether largest count found
            if(largest_count < count):
                largest_count = count
            count = 0 

    print(largest_count)
    return largest_count


def vert_line_counter(img):
    filtered_img = img[:,:,1]
    print(img)
    transpose_img = cv2.transpose(filtered_img)

    arraypos = 0
    vert_line_arr = []
    for line in transpose_img:
        if(count_longest_line(line) > len(line)/4): #if the vertical line depth is larger than half the height of the image then line is found
            #store line in array
            vert_line_arr.append(line)
            # img = cv2.line(img,(0,0),(1080,1080),color=(255, 0, 0), thickness=10)
            cv2.line(img,(arraypos,0),(arraypos,len(line)-1),color=(255, 0, 0), thickness=1)
            print(line)
            print(arraypos)
            #
        arraypos = arraypos + 1

    cv2.imshow("PIC",img)
    cv2.waitKey(0) 
  
    #closing all open windows 
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()
