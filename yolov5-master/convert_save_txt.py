import os
import shutil

labels = os.listdir('./runs/detect/exp54/labels/')
outfile = open('./outfile.csv',"w")
for label in labels:
    f = open(f"./runs/detect/exp54/labels/{label}","r")

    line = f.read()
    print(line)
    split = line.split(" ")
    output_string = f"{label[8:-4]},{split[0]},{split[1]},{split[2]},{split[3]},{split[4]},{split[5]}"
    outfile.write(output_string)

outfile.close()
