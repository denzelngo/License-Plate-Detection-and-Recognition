import cv2
import torch
import torch.backends.cudnn as cudnn
import time
import numpy as np
from yolov4_utils.darknet2pytorch import Darknet
from yolov4_utils.torch_utils import do_detect
from queue import Queue
from threading import Thread
from crnet_pytorch_new import CRNet


def lp_recognition(crnet, frame0, box):
    with torch.no_grad():
        x1 = int(box[0] * frame0.shape[1])
        y1 = int(box[1] * frame0.shape[0])
        x2 = int(box[2] * frame0.shape[1])
        y2 = int(box[3] * frame0.shape[0])

        lp_patch = frame0[y1:y2, x1:x2]
        lp = cv2.resize(lp_patch, (352, 128))
        lp = np.array(lp)[:, :, ::-1].transpose(2, 0, 1) / 255.
        lp = torch.Tensor(lp).unsqueeze(0).cuda().half()
        res = crnet(lp, batch=False)
        box_, conf_, cls_ = res
        # Sort the character from left to right
        x1_character = box_[:, 0]
        _, idx = x1_character.sort(0)
        lp_str = ''.join([letter[i] for i in cls_[idx]])
    return x1, y1, x2, y2, lp_str


def detect():
    cudnn.benchmark = True
    cudnn.enabled = True
    # Initialize
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    half = device != 'cpu'  # half precision only supported on CUDA

    # Load models
    # LP detector (YoloV4-tiny)
    cfgfile = 'yolov4_tiny/yolov4_tiny_384x384.cfg'
    lp_detector = Darknet(cfgfile)
    lp_detector.print_network()
    lp_detector.load_weights('yolov4_tiny/yolov4-tiny-custom-anchors-384_best.weights')
    lp_detector.cuda()
    lp_detector.eval()

    # LP characters recognition model (CR-Net, modified YoloV2)
    cr_net = CRNet()
    cr_net.load_weights('./data/ocr_new/lp-recognition.weights')
    cr_net.eval()
    cr_net.half().cuda()
    # Load alphabet and number
    global letter
    with open('data/ocr/ocr-net.names') as f:
        letter = f.read().splitlines()

    # Start inference
    # video = 'test_2.mp4'
    video = 'videos/video_1.mp4'
    video = 0
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('Video FPS: ', fps)
    cv2.namedWindow('LP detection & recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('LP detection & recognition', 800, 800)

    save_img = False
    if save_img:
        fourcc = 'mp4v'  # output video codec

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter('out_pxl_3_ocr_new_y4t.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    count = 0
    q_data = Queue()
    while cap.isOpened():
        count += 1
        print('----------------')
        print('Frame ', count)
        ret, frame0 = cap.read()
        if not ret:
            break
        frame0_h, frame0_w, _ = frame0.shape
        if 'PXL' in str(video):
            frame0 = cv2.flip(frame0, 0)
            frame0 = cv2.flip(frame0, 1)
        t1 = time.time()

        img_resized = cv2.resize(frame0, (lp_detector.width, lp_detector.height))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        boxes = do_detect(lp_detector, img_rgb, conf_thresh=0.6, nms_thresh=0.4, use_cuda=True)
        for i in range(len(boxes[0])):
            box = boxes[0][i]
            t = Thread(
                target=lambda q, arg1, arg2, arg3: q.put(lp_recognition(arg1, arg2, arg3)),
                args=(q_data, cr_net, frame0, box))
            t.isDaemon()
            t.start()
            t.join()
        for k in range(q_data.qsize()):
            x1, y1, x2, y2, lp_str = q_data.get()

            font = cv2.FONT_HERSHEY_SIMPLEX
            wh_text = cv2.getTextSize(lp_str, font, 2, 5)[0]
            pos2 = (x1 + wh_text[0]), (y1 - wh_text[1] - 3)
            pos1 = (x1, y1)
            cv2.rectangle(frame0, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(frame0, pos1, pos2, (255, 255, 255), -1)
            cv2.putText(frame0, lp_str, (x1, y1 - 2), font, 2, (0, 0, 0), 5)

        t2 = time.time()
        fps2 = 1 / (t2 - t1)
        print('Total FPS: ', fps2)
        frame0 = cv2.putText(frame0, 'FPS {:.2f}'.format(fps2), (20, 50),
                             cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 6)
        cv2.imshow('LP detection & recognition', frame0)
        if save_img:
            vid_writer.write(frame0)
        if cv2.waitKey(1) == 27:  # q to quit
            break


if __name__ == '__main__':
    with torch.no_grad():
        detect()
