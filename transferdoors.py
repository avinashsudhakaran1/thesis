import os
import shutil

labels = os.listdir('./labels')

for label in labels:
    shutil.copy('./Deeplabv3_custom/Doors/Images/'+label , './NewDoors/'+label)