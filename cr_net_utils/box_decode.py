import torch
from cr_net_utils.anchorBox import AnchorBox
from cr_net_utils.detect import box_to_corners, nms, filter_box


def get_boxes(out):
    res = []
    nms_t, score_t = 0.4, 0.4
    anchor_box = AnchorBox().forward()
    b, c, h, w = out.size()  # [N,80,10,30]
    feat = out.permute(0, 2, 3, 1).contiguous().view(b, -1, 2, 35 + 5)

    box_xy, box_wh = torch.sigmoid(feat[..., 0:2]), feat[..., 2:4].exp()
    box_conf, score_pred = torch.sigmoid(feat[..., 4:5]), feat[..., 5:].contiguous()
    box_prob = torch.softmax(score_pred, dim=3)
    box_pred = torch.cat([box_xy, box_wh], 3)
    width, height = (240, 80)
    img_shape = torch.Tensor([[width, height, width, height]])
    anchor_box, img_shape = anchor_box.cuda(), img_shape.cuda()
    anchor_box = anchor_box.view_as(box_pred[0]).expand(box_pred.size())
    box_pred[..., 0:2] += anchor_box[..., 0:2]
    box_pred[..., 2:] *= anchor_box[..., 2:]
    boxes = box_to_corners(box_pred)
    boxes[..., 0] /= 30
    boxes[..., 1] /= 10
    boxes[..., 2] /= 30
    boxes[..., 3] /= 10

    for i, box in enumerate(boxes):

        boxes, scores, classes = filter_box(box.unsqueeze(0), box_conf[i].unsqueeze(0), box_prob[i].unsqueeze(0),
                                            score_t)
        if boxes.numel() == 0: continue
        boxes = boxes * img_shape.repeat(boxes.size(0), 1)
        keep, count = nms(boxes, scores, nms_t)
        boxes = boxes[keep[:count]]
        scores = scores[keep[:count]]
        classes = classes[keep[:count]]
        ind = i
        res.append((boxes, scores, classes, ind))

    return res
