import os
import shutil

labels = os.listdir('./yolov5labels')

for label in labels:
    shutil.copy(f'./Deeplabv3_custom/inputs/Doors/Images/{label[:-4]}.png' , f'./yolov5doors/{label[:-4]}.png')