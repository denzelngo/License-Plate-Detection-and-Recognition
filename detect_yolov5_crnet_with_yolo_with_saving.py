from models.experimental import *
from yolov5_utils.datasets import *
from yolov5_utils.utils import *
from crnet_pytorch_with_yolo import CRNet
from queue import Queue
from threading import Thread
from difflib import SequenceMatcher
from pathlib import Path


def nms_crnet(prediction, conf_thres=0.4, iou_thres=0.2):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    xc = prediction[..., 4] > conf_thres  # candidates

    x = prediction[xc]  # confidence

    # If none remain process next image
    if not x.shape[0]:
        return None

    # Box (center x, center y, width, height) to (x1, y1, x2, y2)
    box = xywh2xyxy(x[:, :4])

    n = x.shape[0]  # number of boxes
    if not n:
        return None

    # Sort by confidence
    # x = x[x[:, 4].argsort(descending=True)]

    # Batched NMS
    scores = x[:, 4]

    cls = x[:, 5]
    i = torchvision.ops.boxes.nms(box, scores, iou_thres)
    output = torch.cat((box[i], scores[i].view(-1, 1), cls[i].view(-1, 1)), -1)

    return output


def update_plate_patch(old_plate, new_plate, score):
    if old_plate[0] is None:
        old_plate[0] = new_plate
        old_plate[1] = score
    else:
        if score > old_plate[1]:
            old_plate[0] = new_plate
            old_plate[1] = score


def check_and_edit_lp(lp_str_tuple, lp_dict):
    lp_str, lp_count = lp_str_tuple
    str_to_check = lp_str[:2] + lp_str[-2:]  # the first 2 letters and the last 2 letters

    if str_to_check.isalpha() and len(lp_str) == 7:
        return lp_str, lp_count
    count_alpha = [i for i in str_to_check if i.isalpha()]  # count the alphabet letter in lp_str
    if len(count_alpha) >= 2 and len(lp_str) == 7:
        # New type French LP (7 letters, AA-111-AA format)
        if '0' in str_to_check:  # check if 0 appears in the positions of alphabet

            # Pop this wrong lp_str
            lp_dict.pop(lp_str)
            if len(lp_dict):
                lp_str_tuple = max(lp_dict.items(), key=lambda x: x[1])
                lp_str, lp_count = lp_str_tuple
            else:
                return None
        str_to_check = lp_str[:2] + lp_str[-2:]  # the first 2 letters and the last 2 letters
        if '8' in str_to_check:  # switch 8 to B
            index = [i for i, x in enumerate(lp_str) if x == '8' and i not in [2, 3, 4]]
            for i in index:
                lp_str_list = list(lp_str)
                lp_str_list[i] = 'B'
            lp_str = "".join(lp_str_list)

        if 'B' in lp_str[2:5]:  # switch B to 8
            index = [i for i, x in enumerate(lp_str) if x == 'B' and i not in [0, 1, 5, 6]]
            for i in index:
                lp_str_list = list(lp_str)
                lp_str_list[i] = '8'
            lp_str = "".join(lp_str_list)

        return lp_str, lp_count
    else:
        #  Old type French LP (4-8 letters, 111-AAA-00 format)
        return lp_str, lp_count



def lp_recognition(args):
    global new_plate_count
    crnet, frame0, xyxy, conf, plates_dict = args
    with torch.no_grad():
        x1, y1, x2, y2 = xyxy
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        plate_shape = (352, 128)

        lp_patch = frame0[y1:y2, x1:x2]
        # lp_patch = cv2.copyMakeBorder(lp_patch, 20, 20, 0, 0, cv2.BORDER_CONSTANT, (114, 114, 114))  # tryyyyy
        lp_ori = cv2.resize(lp_patch, plate_shape)
        lp = np.array(lp_ori.copy())[:, :, ::-1].transpose(2, 0, 1) / 255.
        lp = torch.tensor(lp, dtype=torch.float).unsqueeze(0).cuda()
        res = crnet(lp, plate_shape)[0]
        res = nms_crnet(res)
        if res is not None:
            # Sort the character from left to right
            x1_character = res[:, 0]
            cls = res[:, -1]
            _, idx = x1_character.sort(0)
            lp_str = ''.join([letter[int(i)] for i in cls[idx]])
            if len(lp_str) > 5:
                lp_assigned = False
                for plate, plate_info in plates_dict.items():
                    string_dict = plate_info['string']
                    plate_img = plate_info['plate_patch']
                    if string_dict:
                        if lp_str in string_dict:
                            string_dict[lp_str] += 1
                            lp_assigned = True
                            update_plate_patch(plate_img, lp_ori, conf)
                            break
                        else:
                            max_conf_string = max(string_dict.items(), key=lambda x: x[1])[0]
                            similarity = SequenceMatcher(None, max_conf_string, lp_str).ratio()
                            if similarity > 0.6:
                                string_dict[lp_str] = 0
                                lp_assigned = True
                                update_plate_patch(plate_img, lp_ori, conf)
                                break
                    else:
                        string_dict[lp_str] = 0
                        lp_assigned = True
                        update_plate_patch(plate_img, lp_ori, conf)
                        break
                if not lp_assigned:
                    new_plate_count += 1
                    new_plate_key = 'plate_' + str(new_plate_count)
                    plates_dict[new_plate_key] = {'string': {lp_str: 0}, 'plate_patch': [lp_ori, conf]}

            return x1, y1, x2, y2, lp_str
        else:
            return x1, y1, x2, y2, None


def detect():
    global new_plate_count
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
    cr_net.to('cuda')
    # Load alphabet and number
    with open('data/ocr/ocr-net.names') as f:
        letter = f.read().splitlines()

    new_plate_count = 0

    # Start inference
    # video = 'test_2.mp4'
    video = 'videos/PXL_5.mp4'
    # video = 0
    cap = cv2.VideoCapture(video)
    cv2.namedWindow('Vehicle detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Vehicle detection', 600, 600)

    fps1 = 0
    save_img = True
    if save_img:
        fourcc = 'mp4v'  # output video codec
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter('out_PXL_5.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    count = 0
    q_data = Queue()

    # Dict to store info of plates
    plates_dict = {'plate_0': {'string': {}, 'plate_patch': [None, None]}}

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
                    args = cr_net, frame0, xyxy, conf, plates_dict
                    t = Thread(target=lambda q, arg: q.put(lp_recognition(arg)), args=(q_data, args))

                    t.isDaemon()
                    t.start()
                    t.join()
                for k in range(q_data.qsize()):
                    x1, y1, x2, y2, lp_str = q_data.get()
                    if lp_str is not None:
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
        frame0 = cv2.putText(frame0, 'FPS {:.2f}'.format(fps2), (20, 50),
                             cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 6)
        # frame0 = cv2.resize(frame0, (int(frame0_w / 3), int(frame0_h / 3)))
        cv2.imshow('Vehicle detection', frame0)
        if save_img:
            vid_writer.write(frame0)
        if cv2.waitKey(1) == 27:  # q to quit
            break
    save_path = Path('output') / 'PXL_5_vuidua'
    save_path.mkdir(parents=True, exist_ok=True)

    for plate_key, plate_info in plates_dict.items():
        string_dict = plate_info['string']
        plate_img = plate_info['plate_patch'][0]
        print(plate_key, string_dict)
        max_conf_string = max(string_dict.items(), key=lambda x: x[1])
        # Switch processing
        max_conf_string = check_and_edit_lp(max_conf_string, string_dict)
        if max_conf_string is not None:
            if max_conf_string[1] >= 2:
                save_img = save_path / ('{}.png'.format(max_conf_string[0]))
                cv2.imwrite(str(save_img), plate_img)

    print('Total number of plates: ', len(plates_dict))


if __name__ == '__main__':
    with torch.no_grad():
        detect()
