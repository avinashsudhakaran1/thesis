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
from pathlib import Path
from torch import nn, optim
from torchvision import datasets, transforms
import zipfile
import shutil

def main():
    model = load_model()
    
    # Specify SVO path parameter
    svo_input_path = r"D:/ZED data/doorway.svo"
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
    svo_position = 200
    frame_count = 0 # to count total frames
    total_fps = 0 # to get the final frames per second
    img_array = []
    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()
            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)
            src_img = left_image.get_data()
            
            # get the start time
            start_time = time.time()

            detect(model,src_img)

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
            
    #closing all open windows 
    cv2.destroyAllWindows()
    zed.close()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}") 
    return 0

device = ("cuda" if torch.cuda.is_available() else "cpu") # Use GPU or CPU for training
class_names = ['closed','open']
######     MODEL     ###########
def load_model():
    model = AlexNet()
    model = model.to(device)
    model.load_state_dict(torch.load("C:/Users/Fred/Desktop/Thesis/AlexNet/100epochs.pt"))
    model.eval()
    return model


def detect(model,img):
    #transpose to rgb image format 
    img = img[:,:,:3].transpose(2,0,1)
    img = img.reshape(1,3,1080,1920)

    # transform = transforms.Compose([
    #       transforms.ToPILImage(),
    #       transforms.Resize((360,640)),
    #     #   transforms.Lambda(lambda x: x.to(device)),
    #       transforms.ToTensor()
    #       ])
    # tsfm = transform(img)
    # tsfm = tsfm.to(device)

    with torch.no_grad():
        # images = tsfm.to(device)
        # pred = model(img)
        pred = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255) #/255 makes array of numbers from 0 to 1
    print(torch.max(pred))
    
    # show the image
    # cv2.imshow("Original Image", rgb_image)
    # cv2.imshow("Median Image", median_img_array* np.uint8(255))
    cv2.imshow("Mean Image", img[0][0])
    cv2.waitKey(1)
    # return mean_img_array


def find_blob(segmented_img,rgb_image):
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

    cv2.imshow('out', rgb_image)
    cv2.waitKey(1)

class AlexNet(nn.Module):

    def __init__(self, num_classes: int = 2):
        super(AlexNet, self).__init__()

        self.convolutional = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.linear = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.convolutional(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return torch.softmax(x, 1)

if __name__ == "__main__":
    main()