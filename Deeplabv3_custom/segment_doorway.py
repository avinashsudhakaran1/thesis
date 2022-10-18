from statistics import median
import torch
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import os
import math
import sys
import pyzed.sl as sl
import time

def main():
    model = load_model()
    
    # Specify SVO path parameter
    svo_input_path = r"D:/ZED data/indoor.svo"
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_input_path))
    # init_params.svo_real_time_mode = False  # Don't convert in realtime
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
    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second
    img_array = []
    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()
            print(svo_position)
            if(svo_position > 1115):
                # Retrieve SVO images
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                src_img = left_image.get_data()
                
                start = time.time()

                segmented_img = segment(model,src_img)
                

                post_seg = time.time()
                post_seg_diff = post_seg - start
                print("SEGMENT TIME:   " + str(post_seg_diff))
                
                find_blob(segmented_img,src_img,svo_position)

                blob_time = time.time() 
                blob_time_diff = (blob_time - post_seg)
                print("BLOB TIME:::    " + str(blob_time_diff))
                # get the end time
                end_time = time.time()
                # get the current fps
                fps = 1 / (end_time - start)
                # add current fps to total fps
                total_fps += fps
                # increment frame count
                frame_count += 1
            # Check if we have reached the end of the video
            if svo_position >=3845:#(nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
            
    #closing all open windows 
    cv2.destroyAllWindows()
    zed.close()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}") 
    return 0






# input_folder = "./inputs/70TestDoors"
# output_folder = "C:/Users/Fred/Desktop/Thesis/Deeplabv3_custom/outputs/newdataset_secondtry"
output_folder = "C:/Users/Fred/Desktop/Thesis/Deeplabv3_custom/outputs/20221012_e75b4"

######     MODEL     ###########
def load_model():
    model = torch.load(output_folder + '/weights.pt')
    print("Model::::" + output_folder + 'weights.pt')
    # Set the model to evaluate mode
    model.eval()
    return model


def segment(model,img):
    #transpose to rgb image format 
    img = img[:,:,:3].transpose(2,0,1)
    # img = cv2.resize(img[0][0],(720,1280))
    # img = np.stack((img,)*3, axis=-1)
    # img = img.reshape(1,3,720,1280)
    img = img.reshape(1,3,1080,1920)
    # img = cv2.resize(img[0][0],(360,640)).reshape(1,3,360,640)
    with torch.no_grad():
        a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255) #/255 makes array of numbers from 0 to 1

    median = np.median(a['out'].data.cpu().numpy().flatten())
    mean = a['out'].data.cpu().numpy().flatten().mean()

    # Plot the input image, ground truth and the predicted output
    # rgb_image = cv2.cvtColor(cv2.imread(f'{input_folder}/Images/{door}.png'), cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(f'{input_folder}/Masks/{door}.png')
    median_img_array = a['out'].cpu().detach().numpy()[0][0]> median
    mean_img_array = a['out'].cpu().detach().numpy()[0][0]> mean

    # show the image
    # cv2.imshow("Original Image", rgb_image)
    # cv2.imshow("Median Image", median_img_array* np.uint8(255))
    # cv2.imshow("Mean Image", mean_img_array* np.uint8(255))

    return mean_img_array


def find_blob(segmented_img,rgb_image,svo_position):
    #######     FIND BLOB    ########
    #https://stackoverflow.com/questions/56589691/how-to-leave-only-the-largest-blob-in-an-image 

    # Read image as black and white instead of true or false
    segmented_img_bw = segmented_img * np.uint8(255)
    # Generate intermediate image; use morphological closing to keep parts of the brain together
    inter = cv2.morphologyEx(segmented_img_bw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find largest contour in intermediate image
    contours, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest_c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(biggest_c)
    
    # Output
    out = np.zeros(inter.shape, np.uint8)
    out = cv2.bitwise_and(segmented_img_bw, out)

    #reconfigure rgb image for cv2
    rgb_image = rgb_image[:,:]
    # draw the biggest contour (c) in green
    cv2.drawContours(rgb_image, [biggest_c], -1, 255, cv2.FILLED)
    cv2.rectangle(rgb_image,(x,y),(x+w,y+h),255,10)
    rectangle_area = w*h
    blob_area = cv2.contourArea(biggest_c)

    confidence = (blob_area/rectangle_area) * 100

    # cv2.imshow('out', rgb_image)
    # cv2.waitKey(1)
    # cv2.imwrite(f'C:/Users/Fred/Desktop/Thesis/Deeplabv3_custom/doorway_ambient/door_{svo_position}_{confidence:.2f}.png', rgb_image)


if __name__ == "__main__":
    main()