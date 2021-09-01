import sys
sys.path.append("./SFD_pytorch/")

import cv2
import torch
import numpy as np
import net_s3fd
from bbox import *
from settings import *
from preprocess import preprocess
from models.face_mask_detector import FaceMaskDetector
from torch.autograd import Variable
import torch.nn.functional as F


############################ FROM https://github.com/clcarwin/SFD_pytorch
def detect_face(net,img):
    img = img - np.array([104,117,123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,)+img.shape)

    # img = torch.FloatTensor(img)
    # if CUDA:
    #     img = img.cuda()

    img = Variable(torch.from_numpy(img).float(),volatile=True)
    if CUDA:
        img = img.cuda()

    BB,CC,HH,WW = img.size()
    olist = net(img)

    bboxlist = []
    for i in range(len(olist)//2):
        olist[i*2] = F.softmax(olist[i*2])
    for i in range(len(olist)//2):
        ocls,oreg = olist[i*2].data.cpu(),olist[i*2+1].data.cpu()
        FB,FC,FH,FW = ocls.size() # feature map size
        stride = 2**(i+2)    # 4,8,16,32,64,128
        anchor = stride*4
        for Findex in range(FH*FW):
            windex,hindex = Findex%FW,Findex//FW
            axc,ayc = stride/2+windex*stride,stride/2+hindex*stride
            score = ocls[0,1,hindex,windex]
            loc = oreg[0,:,hindex,windex].contiguous().view(1,4)
            if score<0.05: continue
            priors = torch.Tensor([[axc/1.0,ayc/1.0,stride*4/1.0,stride*4/1.0]])
            variances = [0.1,0.2]
            box = decode(loc,priors,variances)
            x1,y1,x2,y2 = box[0]*1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1,y1,x2,y2,score])
    bboxlist = np.array(bboxlist)
    if 0==len(bboxlist): bboxlist=np.zeros((1, 5))
    return bboxlist
######################################################################################


def detect_mask(model, image):
    image = cv2.resize(image, dsize=(IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocess(image)
    image = image.transpose(2, 0, 1)
    image = torch.FloatTensor(image)

    if CUDA is True:
        image = image.cuda()

    with torch.no_grad():
        pred = model(image.unsqueeze(0))
        pred = pred.cpu().numpy().squeeze()

    if pred < 0.5:
        print("NO MASK!")
    else:
        print(":)")


def run():
    cap = cv2.VideoCapture(0)
    detector = FaceMaskDetector()
    detector.load_state_dict(torch.load(FACE_MASK_DETECTOR_CKPT))

    net = net_s3fd.s3fd()
    net.load_state_dict(torch.load(FACE_DETECTOR_CKPT))

    if CUDA is True:
        detector = detector.cuda()
        net = net.cuda()
        
    detector.eval()
    net.eval()
    
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print(f"Some error in reading frame...")
            continue

        orignal_image = frame.copy()
        frame = cv2.resize(frame, dsize=(128, 128))
        
        resized_image = frame.copy()
        bboxlist = detect_face(net, frame)

        keep = nms(bboxlist, 0.3)
        bboxlist = bboxlist[keep,:]

        if len(bboxlist) > 0:
            x1, y1, x2, y2, s = bboxlist[0]
            candidate_bbox = [x1, y1, x2, y2]
            score = s

            for b in bboxlist[1:]:
                x1, y1, x2, y2, s = b
                if s > score:
                    candidate_bbox = [x1, y1, x2, y2]
                    score = s

            # print(score)
            if score > 0.5:
                x1, y1, x2, y2 = candidate_bbox
                face = resized_image[int(y1):int(y2), int(x1):int(x2)]
                detect_mask(detector, face)

        cv2.imshow('test', orignal_image)
        key = cv2.waitKey(10)
        if key == 27:
            break


if __name__ == "__main__":
    run()
