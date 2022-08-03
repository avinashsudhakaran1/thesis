import torch
import torchvision
from PIL import Image
from torchvision import transforms
import cv2
import time
import segmentation_utils
import bisenet

# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# download or load the model from disk
# model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
path = "best_dice_loss_miou_0.655.pth"
model = bisenet.BiSeNet(device)
model.load_state_dict(torch.load(path, map_location=device))
# load the model onto the computation device
# model = model.eval().to(device)
# model = model.to(device)
vid = "./goprotest2.mp4"

# sample execution (requires torchvision)
# https://debuggercafe.com/semantic-segmentation-using-pytorch-deeplabv3-resnet50/ 

cap = cv2.VideoCapture(vid)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
save_name = "outputvid"
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"outputs/{save_name}.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second


while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # get predictions for the current frame
            outputs = segmentation_utils.get_segment_labels(rgb_frame, model, device)
            # outputs = model(rgb_frame)
        
        # obtain the segmentation map
        segmented_image = segmentation_utils.draw_segmentation_map(outputs['out'])
        # segmented_image = segmentation_utils.draw_segmentation_map(outputs['out'][0].argmax(0).byte().cpu().numpy())
        # get the final image with segmentation map overlayed on original iimage
        final_image = segmentation_utils.image_overlay(rgb_frame, segmented_image)
        # get the end time
        end_time = time.time()
        # get the current fps
        fps = 1 / (end_time - start_time)
        # add current fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # put the FPS text on the current frame
        cv2.putText(final_image, f"{fps:.3f} FPS", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # press `q` to exit
        cv2.imshow('image', final_image)
        out.write(final_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}") 