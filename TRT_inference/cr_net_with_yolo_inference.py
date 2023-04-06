
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time
import cv2
import numpy as np
import argparse
# from cr_net_post_processing import process

from camera import add_camera_args, Camera
INPUT_W = 352
INPUT_H = 128
CONF_THRESH = 0.4
IOU_THRESHOLD = 0.2
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
        c2 = c1[0] + t_size[0], c1[1] + t_size[1] + 4
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0]  , c1[1] + t_size[1] +1),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
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

def xywh2xyxy_old(origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes tensor, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes tensor, each row is a box [x1, y1, x2, y2]
        """
        #y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
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
        #y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y = np.zeros_like(x)
        r_w = INPUT_W / origin_w
        r_h = INPUT_H / origin_h
        y[:, 0] = (x[:, 0] - x[:, 2] / 2)/r_w
        y[:, 2] = (x[:, 0] + x[:, 2] / 2)/r_w
        y[:, 1] = (x[:, 1] - x[:, 3] / 2)/r_h 
        y[:, 3] = (x[:, 1] + x[:, 3] / 2)/r_h 


        return y


def post_process(pred, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            pred:     A tensor likes [[cx,cy,w,h,conf,cls_id], [cx,cy,w,h,conf,cls_id], ...]] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes tensor, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a tensor, each element is the score correspoing to box
            result_classid: finally classid, a tensor, each element is the classid correspoing to box
        """

        # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4:5]
        # print(scores.shape)
        # print(scores[(-scores).argsort(0),:][:50])

        # Get the classid
        classid = pred[:, 5]
        # Choose those boxes that score > CONF_THRESH
        si = scores.flatten() > CONF_THRESH
        boxes = boxes[si, :]
        scores = scores[si]
        classid = classid[si]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes = xywh2xyxy_old(origin_h, origin_w, boxes)
        boxes_scores = np.concatenate((boxes,scores),axis=1)
        
        # Do nms
        indices = nms(boxes_scores,threshold=IOU_THRESHOLD)
        #indices = torchvision.ops.nms(boxes, scores, iou_threshold=IOU_THRESHOLD).cpu()
        result_boxes = boxes[indices, :]
        result_scores = scores[indices].flatten()
        result_classid = classid[indices]
        return result_boxes, result_classid, result_scores
        #return boxes, scores, classid
def test_output(pred):
            # Get the boxes
        boxes = pred[:, :4]
        # Get the scores
        scores = pred[:, 4:5]
        idx = (-scores).argsort(0)
        # print(scores.shape)
        # print(scores[(-scores).argsort(0),:][:50])

        # Get the classid
        classid = pred[:, 5]

        return classid[idx], scores[idx]


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


def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host
    # inputs/outputs.
    size0 = engine.get_binding_shape(0)
    dtype0 = engine.get_binding_dtype(0)
    # print('dtype0: ', type(dtype0))
    size1 = engine.get_binding_shape(1)
    dtype1 = engine.get_binding_dtype(1)
    # print('dtype1: ', type(dtype1))
    h_input = cuda.pagelocked_empty(trt.volume(size0), dtype=trt.nptype(dtype0))
    h_output = cuda.pagelocked_empty(trt.volume(size1), dtype=trt.nptype(dtype1))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


tick = time.time()
args = parse_args()
args.onboard = 0

with open('ocr-net.names') as f:
    letter = f.read().splitlines()

#output_shape = (1,200,16,44)
output_shape = (1500,6)
anchors = [(0.7685, 1.2664), (0.5706, 1.8263), (0.9809, 1.6286), (1.1587, 1.9536), (1.3615, 2.3898)]
#cam = Camera(args)
fps_list = []
with open('engine/cr_net_240x80_with_yolo.engine', 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.INFO)) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
    #print(f'Model loaded. ({time.time() - tick} s)')
    tick = time.time()
    h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
    #print(f'Model allocated. ({time.time() - tick} s)')
    img0 = cv2.imread('/home/anas/Downloads/CL9650L.png')

    with engine.create_execution_context() as context:
        for i in range(10):
            #img0 = cam.read()
            
            H0, W0, _ = img0.shape 
            # print('img shape: ', H0, W0)

            #if img0 is None:
            #         break
            #img_height, img_width, _ = img0.shape
            #img = imageio.imread('no_mask_example.png')
            img = cv2.resize(img0, (240, 80))
            # img = cv2.resize(img0, (352, 128))
            #img = img.reshape(-1, 128, 128, 3)
            img = np.asarray(img)[:, :, ::-1].transpose((2, 0, 1))/255.
            img = np.asarray(img).astype(np.float32).ravel()
            #img = img/255.0
            np.copyto(h_input, img)

            
            tick2 = time.time()
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(d_input, h_input, stream)
            # Run inference.
            context.execute_async_v2(bindings=[int(d_input), int(
                d_output)], stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(h_output, d_output, stream)
            # Synchronize the stream
            stream.synchronize()

            
            

            output = h_output.reshape(output_shape)
            # test_id, test_score = test_output(output)
            # print(test_id[:20])
            # print(test_score[:20])
            boxes, categories, conf = post_process(output, H0, W0)
            # print(boxes)

            x1 = boxes[:, 0]
            idx = x1.argsort(0)
            lp_str = ''.join([letter[int(i)] for i in categories[idx]])

            fps = 1/(time.time()-tick2)
            fps_list.append(fps)

            # print(output)
            #print('Plate number: ', lp_str, fps)
        
        color = (0,0,255)
        for i in range(len(boxes)):
            box = boxes[i]
            label = letter[int(categories[i])]
            plot_one_box(
                box,
                img0,
                color,
                label=label)
        cv2.imwrite('out_put.jpg', img0)

        



            # tick3 = time.time()
            # boxes, categories, conf = process(output, anchors, np.array([W0, H0]))
            
            

            # x1 = boxes[:, 0]

            # idx = x1.argsort(0)
            # lp_str = ''.join([letter[i] for i in categories[idx]])

            # fps2 = 1/(time.time()-tick3)



            # print('Plate number: ', lp_str,fps2)
            # # print('Result: ',str(output.shape))
            # # print('Result: ',output)



print('Plate number: ', lp_str)
print(fps_list)
print('FPS: ', np.mean(fps_list[3:]))
#cam.release()
cv2.destroyAllWindows()

