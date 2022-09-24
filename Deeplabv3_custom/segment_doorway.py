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

def main():
    model = load_model()
    
    # Specify SVO path parameter
    svo_input_path = r"C:/Users/Fred/Desktop/Thesis/doorway.svo"
    init_params = sl.InitParameters()
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

    img_array = []
    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            src_img = left_image.get_data()

            median_img = segment(model,src_img)
            find_blob(median_img)
            
            # Check if we have reached the end of the video
            if svo_position >= (nb_frames - 1):  # End of SVO
                sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
                break
            
    #closing all open windows 
    cv2.destroyAllWindows()
    zed.close()
    return 0






# input_folder = "./inputs/70TestDoors"
output_folder = "C:/Users/Fred/Desktop/Thesis/Deeplabv3_custom/outputs/newdataset_secondtry"

######     MODEL     ###########
def load_model():
    model = torch.load(output_folder + '/weights.pt')
    print("Model::::" + output_folder + 'weights.pt')
    # Set the model to evaluate mode
    model.eval()
    # Read the log file using pandas into a dataframe
    # df = pd.read_csv(output_folder + '/log.csv')
    ### Training and testing loss, f1_score and auroc values for the model trained on the CrackForest dataset
    # Plot all the values with respect to the epochs
    # df.plot(x='epoch',figsize=(15,8))
    # print(df[['Train_auroc','Test_auroc']].max())
    return model


def segment(model,img):
    ### Sample Prediction
    # door = 'Door' + '1800'
    # print(f'{input_folder}/Masks/{door}.png')

    # # Read  a sample image from the data-set
    # img = cv2.imread(f'{input_folder}/Images/{door}.png').transpose(2,0,1).reshape(1,3,640,480)
    img = img[:,:,:3].transpose(2,0,1).reshape(1,3,1080,1920)
    with torch.no_grad():
        a = model(torch.from_numpy(img).type(torch.cuda.FloatTensor)/255) #/255 makes array of numbers from 0 to 1
    # Plot histogram of the prediction to find a suitable threshold. From the histogram a 0.1 looks like a good choice.
    plt.hist(a['out'].data.cpu().numpy().flatten())

    median = np.median(a['out'].data.cpu().numpy().flatten())
    print(median)
    mean = a['out'].data.cpu().numpy().flatten().mean()
    print(mean)


    # Plot the input image, ground truth and the predicted output
    # rgb_image = cv2.cvtColor(cv2.imread(f'{input_folder}/Images/{door}.png'), cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(f'{input_folder}/Masks/{door}.png')
    median_img_array = a['out'].cpu().detach().numpy()[0][0]> median
    mean_img_array = a['out'].cpu().detach().numpy()[0][0]> mean

    # show the image
    # cv2.imshow("Original Image", rgb_image)
    # cv2.imshow("Median Image", median_img_array* np.uint8(255))
    # cv2.imshow("Mean Image", mean_img_array* np.uint8(255))

    return median_img_array


def find_blob(median_img_array):
    #######     FIND BLOB    ########
    #https://stackoverflow.com/questions/56589691/how-to-leave-only-the-largest-blob-in-an-image 

    # Read image as black and white instead of true or false
    input_img = median_img_array * np.uint8(255)
    # Generate intermediate image; use morphological closing to keep parts of the brain together
    inter = cv2.morphologyEx(input_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    # Find largest contour in intermediate image
    contours, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    biggest_c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(biggest_c)

    # Output
    out = np.zeros(inter.shape, np.uint8)
    out = cv2.bitwise_and(input_img, out)
    # draw the biggest contour (c) in green
    cv2.drawContours(out, [biggest_c], -1, 255, cv2.FILLED)
    cv2.rectangle(out,(x,y),(x+w,y+h),255,10)

    cv2.imshow('out', out)
    cv2.waitKey(1)


if __name__ == "__main__":
    main()