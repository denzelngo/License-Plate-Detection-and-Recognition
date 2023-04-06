# -*- coding:utf-8 -*-
import torch
from itertools import product


class AnchorBox(object):
    def __init__(self):
        super(AnchorBox, self).__init__()
        self.anchor = [[3.638, 5.409], [3.281, 4.764]]

    def forward(self):
        anchors = []
        anchor_num = len(self.anchor)
        for i, j in product(range(10), range(30)):
            cx = j
            cy = i
            for k in range(anchor_num):
                cw = self.anchor[k][0]
                ch = self.anchor[k][1]
                anchors += [cx, cy, cw, ch]
        anchors = torch.Tensor(anchors).view(-1, 4)
        return anchors


class AnchorBox_new(object):
    def __init__(self):
        super(AnchorBox_new, self).__init__()
        self.anchor = [[0.7685, 1.2664], [0.5706, 1.8263], [0.9809, 1.6286], [1.1587, 1.9536], [1.3615, 2.3898]]

    def forward(self):
        anchors = []
        anchor_num = len(self.anchor)
        for i, j in product(range(16), range(44)):
            cx = j
            cy = i
            for k in range(anchor_num):
                cw = self.anchor[k][0]
                ch = self.anchor[k][1]
                anchors += [cx, cy, cw, ch]
        anchors = torch.Tensor(anchors).view(-1, 4)
        return anchors


if __name__ == '__main__':
    priorbox = AnchorBox()
    print(priorbox.forward())
