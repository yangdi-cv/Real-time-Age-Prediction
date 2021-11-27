# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
from SSR_Network import SSRNet
import torch
import torch.nn as nn
import torchvision.transforms as T
import os
from PIL import Image
from utils import check_duplicate

# CUDA settings
def setup_cuda():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Setting seeds (optional)
    seed = 50
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    return device

device = setup_cuda()

model_file = 'model_SSRNet_batch40_epoch148.pth'
model = SSRNet(image_size=64)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(model_file))
model.eval()

transforms = T.Compose([
    # Resize
    T.Resize((64, 64), interpolation=Image.BILINEAR),
    # Convert to a Pytorch tensor (normalization also included)
    T.ToTensor()
])

prototxtPath = os.path.join("deploy.prototxt.txt")
weightsPath = os.path.join("res10_300x300_ssd_iter_140000.caffemodel")

# load our serialized model from disk
net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

# initialize the video stream and allow the cammera sensor to warmup
vs = VideoStream(src=0).start()
time.sleep(2.0)

pre_ages = []
pre_detections = []
start_time = time.time()
counter = 0
fpslist = []
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=640)

    # grab the frame dimensions and convert it to a blob
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()

    cur_idx = []

    counter += 1

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.5:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype("int")
        height = endY - startY
        width = endX - startX
        if height > width:
            pad = int((height - width) / 2)
        else:
            pad = 15
        # Ensure the cropped face image is square
        image = frame[startY: endY, startX-pad: endX+pad]
        number = image.size
        if number == 0:
            print('Face detection')
        else:
            cv2.imwrite("face_detected.jpg", image)

        face = Image.open("face_detected.jpg")

        # 5.3. Apply the transformations
        face = transforms(face)
        # Expand the tensor along the first dimension
        face = face.unsqueeze(0).to(device)
        pred_age = model(face)
        pred_age = pred_age.item()

        idx = check_duplicate(box, pre_detections)
        if idx is None:  # if the current detection is a new one (not exist in the list)
            pre_detections.append(
                box)  # 'pre_detections' is a list whose each element is a Numpy array of size (1, 4) as box coordinates
            pre_ages.append([
                pred_age])  # 'pre_ages' is a list whose each element is also a list for storing all previous predicted ages
            cur_idx.append(len(pre_detections) - 1)  # store the index of the current detection

        else:  # otherwise, we update the entries in the lists
            pre_detections[idx] = box
            pre_ages[idx].append(pred_age)
            pred_age = np.mean(pre_ages[idx])  # average all previous predictions
            cur_idx.append(idx)  # store the index of the current detection

        # draw the bounding box of the face along with the associated
        # probability
        text = "Pred Age: {:.0f}".format(pred_age)
        y = startY - 5
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0,69,255), 2)
        cv2.rectangle(frame, (startX-2, y-30), (startX+210, startY), (0,69,255), thickness=-1)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        # Frames Per Second (FPS)
        if (time.time() - start_time) != 0:
            fps = counter / (time.time() - start_time)
            fpslist.append(fps)
            fps = np.mean(fpslist)
            cv2.putText(frame, "FPS {0}".format(fps), (400, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            # print("FPS: ", counter / (time.time() - start_time))
            counter = 0
            start_time = time.time()

        # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()