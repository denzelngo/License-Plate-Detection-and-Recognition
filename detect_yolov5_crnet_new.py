from models.experimental import *
from yolov5_utils.datasets import *
from yolov5_utils.utils import *
from wpod_utils.torch_utils import detect_lp, reconstruct_torch
from wpod_utils.label import Shape, writeShapes
from wpod_utils.utils import im2single
from wpod_utils.drawing_utils import draw_losangle, write2img
from crnet_pytorch_new import CRNet
from queue import Queue
from threading import Thread


def lp_recognition(crnet, frame0, xyxy):
    with torch.no_grad():
        x1, y1, x2, y2 = xyxy
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

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
    global letter
    # Initialize
    device = torch_utils.select_device()
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load models

    # LP detector (YoloV5)
    lp_detector = attempt_load('data/lp_detector_yolov5/best.pt', map_location=device)  # load FP32 model
    if half:
        lp_detector.half()  # to FP16

    # LP characters recognition model (CR-Net, modified YoloV2)
    cr_net = CRNet()
    cr_net.load_weights('./data/ocr_new/lp-recognition.weights')
    cr_net.eval()
    cr_net.half().cuda()
    # Load alphabet and number
    with open('data/ocr/ocr-net.names') as f:
        letter = f.read().splitlines()

    # Start inference
    # video = 'test_2.mp4'
    # video = 'videos/video_1.mp4'
    video = 0
    cap = cv2.VideoCapture(video)
    cv2.namedWindow('Vehicle detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Vehicle detection', 600, 600)

    fps1 = 0
    save_img = False
    if save_img:
        fourcc = 'mp4v'  # output video codec
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter('out_test_2_yolov5_crnetnew.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
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

        # frame0 = cv2.putText(frame0, 'Dect mode', (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 6)
        img = letterbox(frame0, new_shape=384)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time.time()
        pred = lp_detector(img)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.4)

        for i, det in enumerate(pred):
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame0.shape).round()

                print('Number of LP: ', len(det))
                for *xyxy, conf, cls in det:
                    t = Thread(
                        target=lambda q, arg1, arg2, arg3: q.put(lp_recognition(arg1, arg2, arg3)),
                        args=(q_data, cr_net, frame0, xyxy))

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
            else:
                print('Number of vehicles: 0')

        t2 = time.time()
        fps2 = 1 / (t2 - t1)
        print('Total FPS: ', fps2)
        frame0 = cv2.putText(frame0, 'FPS {:.2f}, FPS LP detection {:.2f}'.format(fps2, fps1), (20, 50),
                             cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 6)
        # frame0 = cv2.resize(frame0, (int(frame0_w / 3), int(frame0_h / 3)))
        cv2.imshow('Vehicle detection', frame0)
        if save_img:
            vid_writer.write(frame0)
        if cv2.waitKey(1) == 27:  # q to quit
            break


if __name__ == '__main__':
    with torch.no_grad():
        detect()
