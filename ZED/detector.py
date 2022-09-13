import cv2
import os
import math
import numpy as np

#pic243


def main():
    images = os.listdir('./export3')
    # path = r"C:/Users/Fred/Desktop/Thesis/ZED/export3/left000243.png"
    # path = r"C:/Users/Fred/Desktop/Thesis/ZED/export3/left000000.png"
    # path = r"C:/Users/Fred/Desktop/Thesis/ZED/export3/left000144.png"
    for image in images:
        if 'left' in image:
            path = f"C:/Users/Fred/Desktop/Thesis/ZED/export3/{image}"
            depth_image_name = "depth" + image[4:]
            depth_path = f"C:/Users/Fred/Desktop/Thesis/ZED/export3/{depth_image_name}"
            src_img = cv2.imread(path)
            depth_img = cv2.imread(depth_path)

            line_arr = houghlines(src_img)
            valid_line_arr = find_gaps(line_arr,depth_img)
            drawlines(src_img, valid_line_arr)
    #closing all open windows 
    cv2.destroyAllWindows()


def drawlines(img, arr):
    for line in arr:
        pt1_l = line[0]
        pt2_l = line[1]

        cv2.line(img, pt1_l, pt2_l, (0,0,255), 3, cv2.LINE_AA)
    
    cv2.imshow("Image with lines", img)
    # cv2.imwrite("./imagewithlines.jpg",src_img)
    cv2.waitKey(1)

valid_line_arr = []
def find_gaps(line_arr,depth_img):
    prev = None

    door_error_thresh = 20
    open_door_thresh = 30

    #sort line array based on the x value of the first tuple (start point x) so that we can scan from left to right
    line_arr.sort(key=lambda tup: tup[0])

    
    if(len(line_arr) > 1): #can only perform gap finding if at least two lines found
        for i,line in enumerate(line_arr):
            if(i > 0): #skip the first line
                pt_1 = line[0] #start of line
                pt_2 = line[1] #end of line

                prev_pt_1 = prev[0] #start of prev line
                prev_pt_2 = prev[1] #end of prev line

                if(abs(pt_1[0] - prev_pt_1[0]) > 200): #if the difference in x values of the two lines are legit
                    #check the depth values here
                    midpoint_cur = ( int((pt_1[0] + pt_2[0])/2) + door_error_thresh , int((pt_1[1] + pt_2[1])/2) )   #move point to the right a bit more
                    midpoint_prev = ( int((prev_pt_1[0] + prev_pt_2[0])/2) - door_error_thresh , int((prev_pt_1[1] + prev_pt_2[1])/2) )   #move point to the left a bit more
                    mid_midpoint = int((midpoint_cur[0] + midpoint_prev[0]) / 2)
                    #filter depth img into single channel
                    filtered_depth_img = depth_img[:,:,1]
                    #find the horizontal line of pixels at the y position of the midpoint of the vert lines
                    depth_arr = filtered_depth_img[midpoint_prev[1]]
                    mid_depth_check = depth_arr[mid_midpoint]
                    left_depth_check = depth_arr[midpoint_prev[0]]
                    right_depth_check = depth_arr[midpoint_cur[0]]
                    avg_front_plane = int((left_depth_check + right_depth_check) / 2)
                    #filter arr into just the width of the door
                    # depth_arr = depth_arr[midpoint_prev[0] : midpoint_cur[0]]

                    if(avg_front_plane - mid_depth_check  > open_door_thresh):
                        #clear old state
                        valid_line_arr.clear()
                        #we have a valid gap so append the lines we want to keep on the image
                        valid_line_arr.append(prev)
                        valid_line_arr.append(line)


            prev = line
    # valid_line_arr = line_arr
    return valid_line_arr


#https://www.delftstack.com/howto/python/opencv-line-detection/#:~:text=To%20detect%20the%20lines%20present,function%20is%20the%20given%20image.
def houghlines(src_img):
    # cv2.imshow('Original Image',src_img)

    dst_img = cv2.Canny(src_img, 20, 250, None, 3)

    lines = cv2.HoughLines(dst_img, 1, np.pi / 180, 150, None, 0, 0)
    line_arr = []
    for i in range(0, len(lines)):
        rho_l = lines[i][0][0]
        theta_l = lines[i][0][1]
        a_l = math.cos(theta_l)
        b_l = math.sin(theta_l)
        x0_l = a_l * rho_l
        y0_l = b_l * rho_l
        pt1_l = (int(x0_l + 1000*(-b_l)), int(y0_l + 1000*(a_l)))
        pt2_l = (int(x0_l - 1000*(-b_l)), int(y0_l - 1000*(a_l)))

        #delta x should always be less than delta y to be a sort of vertical line
        delta_X = abs(pt2_l[0] - pt1_l[0]) 
        delta_Y = abs(pt2_l[1] - pt1_l[1])
        angle_deg = (math.atan2(delta_X, delta_Y) * 180 / math.pi) #angle from vertical axis
        if(angle_deg < 20):
            line_arr.append((pt1_l,pt2_l))

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
