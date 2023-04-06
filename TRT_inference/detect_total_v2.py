import ctypes
import random
import tensorrt as trt
import pycuda.driver as cuda
# import pycuda.autoinit
import time
import cv2
import numpy as np
import imageio
import argparse
from camera import add_camera_args, Camera




INPUT_W = 352
INPUT_H = 352
CONF_THRESH = 0.4
IOU_THRESHOLD = 0.2

INPUT_W_OCR = 352
INPUT_H_OCR = 128


#OCR_OUTPUT_SHAPE = (1,200,10,30)
OCR_OUTPUT_SHAPE = (1500,6)
OCR_ANCHORS = [(0.7685, 1.2664), (0.5706, 1.8263), (0.9809, 1.6286), (1.1587, 1.9536), (1.3615, 2.3898)]


# load coco labels

# load custom plugins
PLUGIN_LIBRARY = "engine/libyololayer.so"
ctypes.CDLL(PLUGIN_LIBRARY)


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time face detection with TrtMtcnn on Jetson '
            'Nano')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument('--minsize', type=int, default=40,
                        help='minsize (in pixels) for detection [40]')
    args = parser.parse_args()
    return args


def nms(boxes, threshold, type='Union'):
    """Non-Maximum Supression

    # Arguments
        boxes: numpy array [:, 0:5] of [x1, y1, x2, y2, score]'s
        threshold: confidence/score threshold, e.g. 0.5
        type: 'Union' or 'Min'

    # Returns
        A list of indices indicating the result of NMS
    """
    if boxes.shape[0] == 0:
        return []
    xx1, yy1, xx2, yy2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.multiply(xx2-xx1+1, yy2-yy1+1)
    sorted_idx = boxes[:, 4].argsort()

    pick = []
    while len(sorted_idx) > 0:
        # In each loop, pick the last box (highest score) and remove
        # all other boxes with IoU over threshold
        tx1 = np.maximum(xx1[sorted_idx[-1]], xx1[sorted_idx[0:-1]])
        ty1 = np.maximum(yy1[sorted_idx[-1]], yy1[sorted_idx[0:-1]])
        tx2 = np.minimum(xx2[sorted_idx[-1]], xx2[sorted_idx[0:-1]])
        ty2 = np.minimum(yy2[sorted_idx[-1]], yy2[sorted_idx[0:-1]])
        tw = np.maximum(0.0, tx2 - tx1 + 1)
        th = np.maximum(0.0, ty2 - ty1 + 1)
        inter = tw * th
        if type == 'Min':
            iou = inter / \
                np.minimum(areas[sorted_idx[-1]], areas[sorted_idx[0:-1]])
        else:
            iou = inter / \
                (areas[sorted_idx[-1]] + areas[sorted_idx[0:-1]] - inter)
        pick.append(sorted_idx[-1])
        sorted_idx = sorted_idx[np.where(iou <= threshold)[0]]
    return pick


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=90,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host
    # inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(
        engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(
        engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


def allocate_buffers_2(engine):
    host_inputs = []
    cuda_inputs = []
    host_outputs = []
    cuda_outputs = []
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(
            binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        cuda_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        # bindings.append(int(cuda_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            host_inputs.append(host_mem)
            cuda_inputs.append(cuda_mem)
        else:
            host_outputs.append(host_mem)
            cuda_outputs.append(cuda_mem)
    stream = cuda.Stream()
    return host_inputs, cuda_inputs, host_outputs, cuda_outputs, stream


def preprocess_image(image_raw):
    """
    description: Read an image from image path, convert it to RGB,
                 resize and pad it to target size, normalize to [0,1],
                 transform to NCHW format.
    param:
        image_raw: input image
    return:
        image:  the processed image
        image_raw: the original image
        h: original height
        w: original width
    """

    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = INPUT_W / w
    r_h = INPUT_H / h
    if r_h > r_w:
        tw = INPUT_W
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((INPUT_H - th) / 2)
        ty2 = INPUT_H - th - ty1
    else:
        tw = int(r_h * w)
        th = INPUT_H
        tx1 = int((INPUT_W - tw) / 2)
        tx2 = INPUT_W - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (114, 114, 114)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    image = np.ascontiguousarray(image)
    return image, image_raw, h, w


def xywh2xyxy(origin_h, origin_w, x):
    """
    description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    param:
        origin_h:   height of original image
        origin_w:   width of original image
        x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
    return:
        y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
    """
    # y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y = np.zeros_like(x)
    r_w = INPUT_W / origin_w
    r_h = INPUT_H / origin_h
    if r_h > r_w:
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2 - (INPUT_H - r_w * origin_h) / 2
        y /= r_w
    else:
        y[:, 0] = x[:, 0] - x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2 - (INPUT_W - r_h * origin_w) / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        y /= r_h

    return y


def post_process(output, origin_h, origin_w, ocr = False):
    """
    description: postprocess the prediction
    param:
        output:     A tensor likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...]
        origin_h:   height of original image
        origin_w:   width of original image
    return:
        result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
        result_scores: finally scores, a tensor, each element is the score correspoing to box
        result_classid: finally classid, a tensor, each element is the classid correspoing to box
    """
    if not ocr:
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # to a torch Tensor
        # pred = torch.Tensor(pred).cuda()
    else:
        pred = output
    # Get the boxes
    boxes = pred[:, :4]
    # Get the scores
    scores = pred[:, 4:5]
    # Get the classid
    classid = pred[:, 5]
    # Choose those boxes that score > CONF_THRESH
    si = scores.flatten() > CONF_THRESH
    boxes = boxes[si, :]
    scores = scores[si]
    classid = classid[si]
    # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
    boxes = xywh2xyxy(origin_h, origin_w, boxes)
    boxes_scores = np.concatenate((boxes, scores), axis=1)

    # Do nms
    indices = nms(boxes_scores, threshold=IOU_THRESHOLD)
    # indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
    result_boxes = boxes[indices, :]
    result_scores = scores[indices].flatten()
    result_classid = classid[indices]
    return result_boxes, result_scores, result_classid
    # return boxes, scores, classid



def main(cam):
    cuda.init()
    dev = cuda.Device(0)
    ctx = dev.make_context()
    tick = time.time()
    frame = 0

    with open('ocr-net.names') as f:
        letter = f.read().splitlines()

    detection_path = 'engine/lp_detector_yolov5s.engine'
    ocr_path = 'engine/cr_net_240x80_with_yolo_fp32.engine'

    trtLogger = trt.Logger(trt.Logger.INFO)
    with open(detection_path, 'rb') as f, open(ocr_path, 'rb') as f2, trt.Runtime(trtLogger) as runtime:
        print('Loading models ...')
        engine_detector = runtime.deserialize_cuda_engine(f.read())
        engine_ocr = runtime.deserialize_cuda_engine(f2.read())

        # print(f'Models loaded. ({time.time() - tick} s)')
        # tick = time.time()
        h_input, d_input, h_output, d_output, stream = allocate_buffers_2(engine_detector)
        h_input2, d_input2, h_output2, d_output2, stream2 = allocate_buffers(engine_ocr)

        print(f'Models loaded and allocated. ({time.time() - tick} s)')

        with engine_detector.create_execution_context() as context, engine_ocr.create_execution_context() as context_ocr:

            if not cam.isOpened():
                raise SystemExit('ERROR: failed to open camera!')

            cv2.namedWindow("ANPR", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("ANPR", 800, 800)

            while 1:
                img = cam.read()
                # img = cv2.flip(img, 0)
                # img = cv2.flip(img, 1)
                if img is None:
                    break
                img_height, img_width, _ = img.shape

                input_image, image_raw, origin_h, origin_w = preprocess_image(img)
                tick = time.time()

                np.copyto(h_input[0], input_image.ravel())

                # Transfer input data to the GPU.
                cuda.memcpy_htod_async(d_input[0], h_input[0], stream)
                # Run inference.
                context.execute_async(bindings=[int(d_input[0]), int(
                    d_output[0])], stream_handle=stream.handle)
                # Transfer predictions back from the GPU.
                cuda.memcpy_dtoh_async(h_output[0], d_output[0], stream)
                # Synchronize the stream
                stream.synchronize()
                # Post-processing with NMS
                output = h_output[0]
                result_boxes, result_scores, result_classid = post_process(
                    output, origin_h, origin_w)

                # Draw rectangles and labels on the original image
                for i in range(len(result_boxes)):
                    box = result_boxes[i]
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])

                    if (int(x1) < 0) | (int(y1) < 0): continue

                    img_patch = image_raw[y1:y2, x1:x2]
                    H0, W0, _ = img_patch.shape 
                    img_patch = cv2.resize(img_patch, (240, 80))
                    img_patch = np.asarray(img_patch)[:, :, ::-1].transpose((2, 0, 1))/255.
                    img_patch = img_patch.astype(trt.nptype(trt.float16)).ravel()




                    np.copyto(h_input2, img_patch)

                    # Transfer input data to the GPU.
                    cuda.memcpy_htod_async(d_input2, h_input2, stream2)
                    # Run inference.
                    context_ocr.execute_async(
                        bindings=[int(d_input2), int(d_output2)], stream_handle=stream2.handle)
                    # Transfer predictions back from the GPU.
                    cuda.memcpy_dtoh_async(h_output2, d_output2, stream2)
                    # Synchronize the stream
                    stream2.synchronize()

                    output = h_output2.reshape(OCR_OUTPUT_SHAPE)
                    boxes, conf, categories = post_process(output, 80, 240, ocr=True)



                    if len(boxes):
                        x1 = boxes[:, 0]
                        idx = x1.argsort(0)
                        lp_str = ''.join([letter[int(i)] for i in categories[idx]])
                        if len(lp_str) == 7:
                            color = (0, 0, 255)
                            plot_one_box(box,image_raw,color,label=lp_str)
                        elif len(lp_str) > 7:
                            lp_str = lp_str[:7]
                            color = (0, 0, 255)
                            plot_one_box(box,image_raw,color,label=lp_str)
                        else:
                            empty = '_'*(7-len(lp_str))
                            lp_str += empty
                            color = (0, 0, 255)
                            plot_one_box(box,image_raw,color,label=lp_str)

                image_raw = cv2.putText(image_raw, 'FPS: {:0.2f}, #plates: {}'.format(1/(time.time() - tick),len(result_boxes)), (20, img_height - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 3)

                cv2.imshow("ANPR", image_raw)
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break

            cam.release()
            cv2.destroyAllWindows()
            ctx.pop()
            del ctx


if __name__ == '__main__':
    args = parse_args()
    # args.video= 'PXL_3.mp4'
    args.onboard = 0
    cam = Camera(args)
    main(cam)
    print('Done')
