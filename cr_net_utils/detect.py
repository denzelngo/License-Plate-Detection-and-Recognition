# coding:utf-8
from __future__ import division
import torch
import torch.nn as nn


def box_to_corners(boxes):
	box_mins = boxes[..., 0:2] - (boxes[..., 2:4] * 0.5)
	box_maxs = boxes[..., 0:2] + (boxes[..., 2:4] * 0.5)
	return torch.cat([box_mins, box_maxs], 3)


def filter_box(boxes, box_conf, box_prob, threshold=.5):

	box_scores = box_conf.repeat(1, 1, 1, box_prob.size(3)) * box_prob
	box_class_scores, box_classes = torch.max(box_scores, dim=3)
	prediction_mask = box_class_scores > threshold
	prediction_mask4 = prediction_mask.unsqueeze(3).expand(boxes.size())

	boxes = torch.masked_select(boxes, prediction_mask4).contiguous().view(-1, 4)
	scores = torch.masked_select(box_class_scores, prediction_mask)
	classes = torch.masked_select(box_classes, prediction_mask)
	return boxes, scores, classes


def nms(boxes, scores, overlap=0.5, top_k=200):
	keep = scores.new(scores.size(0)).zero_().long()
	if boxes.numel() == 0:
		return keep
	x1 = boxes[:, 0]
	x2 = boxes[:, 2]
	y1 = boxes[:, 1]
	y2 = boxes[:, 3]
	area = torch.mul(x2 - x1, y2 - y1)
	v, idx = scores.sort(0)  # sort in ascending order
	idx = idx[-top_k:]  # indices of the top-k largest vals
	xx1 = boxes.new()
	yy1 = boxes.new()
	xx2 = boxes.new()
	yy2 = boxes.new()
	w = boxes.new()
	h = boxes.new()

	count = 0
	while idx.numel() > 0:
		i = idx[-1]
		keep[count] = i
		count += 1
		if idx.size(0) == 1:
			break
		idx = idx[:-1]
		torch.index_select(x1, 0, idx, out=xx1)
		torch.index_select(y1, 0, idx, out=yy1)
		torch.index_select(x2, 0, idx, out=xx2)
		torch.index_select(y2, 0, idx, out=yy2)

		xx1 = torch.clamp(xx1, min=x1[i])
		yy1 = torch.clamp(yy1, min=y1[i])
		xx2 = torch.clamp(xx2, max=x2[i])
		yy2 = torch.clamp(yy2, max=y2[i])
		w.resize_as_(xx2)
		h.resize_as_(yy2)
		w = torch.clamp(xx2 - xx1, min=0.0)
		h = torch.clamp(yy2 - yy1, min=0.0)
		inter = w * h
		rem_areas = torch.index_select(area, 0, idx)
		iou = inter / (area[i] + rem_areas - inter)
		idx = idx[iou.le(overlap)]

	return keep, count


class Detect(nn.Module):
	def __init__(self):
		super(Detect, self).__init__()
		self.nms_t, self.score_t = 0.4, 0.4

	def forward(self, box_pred, box_conf, box_prob, priors, img_shape, max_boxes=10):
		box_pred[..., 0:2] += priors[..., 0:2]
		box_pred[..., 2:] *= priors[..., 2:]
		boxes = box_to_corners(box_pred)
		boxes[..., 0] /= 30
		boxes[..., 1] /= 10
		boxes[..., 2] /= 30
		boxes[..., 3] /= 10

		boxes, scores, classes = filter_box(boxes, box_conf, box_prob, self.score_t)
		if boxes.numel() == 0:
			return boxes, scores, classes
		boxes = boxes * img_shape.repeat(boxes.size(0), 1)
		keep, count = nms(boxes, scores, self.nms_t)
		boxes = boxes[keep[:count]]
		scores = scores[keep[:count]]
		classes = classes[keep[:count]]


		return boxes, scores, classes

class Detect_new(nn.Module):
	def __init__(self):
		super(Detect_new, self).__init__()
		self.nms_t, self.score_t = 0.4, 0.4

	def forward(self, box_pred, box_conf, box_prob, priors, img_shape, max_boxes=10):
		box_pred[..., 0:2] += priors[..., 0:2]
		box_pred[..., 2:] *= priors[..., 2:]
		boxes = box_to_corners(box_pred)
		boxes[..., 0] /= 44
		boxes[..., 1] /= 16
		boxes[..., 2] /= 44
		boxes[..., 3] /= 16

		boxes, scores, classes = filter_box(boxes, box_conf, box_prob, self.score_t)
		if boxes.numel() == 0:
			return boxes, scores, classes
		boxes = boxes * img_shape.repeat(boxes.size(0), 1)
		keep, count = nms(boxes, scores, self.nms_t)
		boxes = boxes[keep[:count]]
		scores = scores[keep[:count]]
		classes = classes[keep[:count]]


		return boxes, scores, classes
