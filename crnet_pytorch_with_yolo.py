import torch
import numpy as np
from torch import nn
import time


class Conv(nn.Module):
    def __init__(self, filter_size, input_size, output_size, stride=1, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=filter_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(output_size),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)


class CRNet(nn.Module):
    def __init__(self, cuda=True):
        super(CRNet, self).__init__()
        self.anchors = [[0.7685, 1.2664], [0.5706, 1.8263], [0.9809, 1.6286], [1.1587, 1.9536], [1.3615, 2.3898]]
        self.num_classes = 35
        self.cuda = cuda

        self.darknet = nn.Sequential(
            Conv(filter_size=3, input_size=3, output_size=32),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv(filter_size=3, input_size=32, output_size=64),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv(filter_size=3, input_size=64, output_size=128),
            Conv(filter_size=1, input_size=128, output_size=64, padding=0),
            Conv(filter_size=3, input_size=64, output_size=128),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            Conv(filter_size=3, input_size=128, output_size=256),
            Conv(filter_size=1, input_size=256, output_size=128, padding=0),
            Conv(filter_size=3, input_size=128, output_size=256),
            Conv(filter_size=3, input_size=256, output_size=512),
            Conv(filter_size=1, input_size=512, output_size=256, padding=0),
            Conv(filter_size=3, input_size=256, output_size=512),
            nn.Conv2d(512, 200, kernel_size=1, padding=0)
        )
        self.detect_layer = YOLOLayer(self.anchors, self.num_classes)

    def count_weights(self):
        weights_count = 0
        model_dict = self.state_dict()
        for k, v in model_dict.items():
            if not 'num_batches_tracked' in str(k):
                weights_count += v.numel()
        return weights_count

    def load_weights(self, weights_path):
        with open(weights_path, 'rb') as weights_file:
            header = np.fromfile(weights_file, dtype=np.int32, count=5)
            weights = np.fromfile(weights_file, dtype=np.float32)
            ptr = 0
            for k in self.darknet:
                if isinstance(k, Conv):
                    conv_layer = k.conv[0]
                    bn_layer = k.conv[1]
                    # Load BN bias, weights, running mean and running variance
                    num_b = bn_layer.bias.numel()
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    # Load conv. weights
                    num_w = conv_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_w)
                    ptr += num_w
                elif isinstance(k, nn.Conv2d):
                    # Last layer
                    last_layer = k
                    # Load bias
                    num_b = last_layer.bias.numel()
                    bn_b = torch.from_numpy(weights[ptr: ptr + num_b]).view_as(last_layer.bias)
                    last_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Load weights
                    num_w = last_layer.weight.numel()
                    conv_w = torch.from_numpy(weights[ptr: ptr + num_w]).view_as(last_layer.weight)
                    last_layer.weight.data.copy_(conv_w)
                    ptr += num_w

        print('[DarkNet backbone] {}/{} weights loaded.'.format(ptr, len(weights)))

    def forward(self, x, img_shape=(352, 128)):
        out = self.darknet(x)
        out = self.detect_layer(out, img_dim=img_shape, cuda=self.cuda)
        return out


class YOLOLayer(nn.Module):
    """Detection layer"""

    def __init__(self, anchors, num_classes, img_dim=(352, 128)):
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes

        self.img_dim = img_dim
        self.grid_size = (0, 0)  # grid size
        self.grid = None

    def compute_grid_offsets(self, grid_size, cuda=True):
        self.grid_size = grid_size

        h_feat, w_feat = grid_size
        print('Feat map: ', h_feat, w_feat)

        col = torch.arange(w_feat).repeat(h_feat).view(-1, w_feat)
        row = torch.arange(h_feat).repeat(w_feat).view(-1, h_feat).t()

        col = col.unsqueeze(2).expand(h_feat, w_feat, self.num_anchors).reshape(1, -1, self.num_anchors, 1)
        row = row.unsqueeze(2).expand(h_feat, w_feat, self.num_anchors).reshape(1, -1, self.num_anchors, 1)
        self.grid = torch.cat((col, row), -1)
        if cuda:
            self.grid = self.grid.cuda()

    def forward(self, x, img_dim=(352, 128), cuda=True):
        # Tensors for cuda support

        self.img_dim = img_dim
        img_w, img_h = img_dim
        image_tensor = torch.tensor([img_w, img_h, img_w, img_h])
        image_tensor = image_tensor.cuda() if cuda else image_tensor

        b, c, h_feat, w_feat = x.size()  # [N,200,16,44]
        grid_size = h_feat, w_feat
        feat = x.permute(0, 2, 3, 1).contiguous().view(b, -1, self.num_anchors, self.num_classes + 5)

        # Get outputs
        box_xy, box_wh = torch.sigmoid(feat[..., 0:2]), feat[..., 2:4].exp()
        box_confidence, box_class_probs = torch.sigmoid(feat[..., 4:5]), torch.sigmoid(feat[..., 5:])
        box_scores = box_confidence * box_class_probs
        box_conf, _ = torch.max(box_scores, -1, keepdim=True)
        box_classes = torch.argmax(box_scores, -1, keepdim=True)

        box_classes = box_classes.view(b, -1, 1)
        box_conf = box_conf.view(b, -1, 1)

        anchors_tensor = torch.tensor(self.anchors).view(1, 1, self.num_anchors, 2)
        if cuda:
            anchors_tensor = anchors_tensor.cuda()
        box_wh *= anchors_tensor

        # If grid size does not match current we compute new offsets
        if grid_size != self.grid_size:
            self.compute_grid_offsets(grid_size, cuda)

        box_xy += self.grid
        # box_xy[..., 0] /= w_feat
        # box_xy[..., 1] /= h_feat
        # box_wh[..., 0] /= w_feat
        # box_wh[..., 1] /= h_feat

        wh_tensor = torch.tensor([w_feat, h_feat]).cuda() if cuda else torch.tensor([w_feat, h_feat])

        box_xy /= wh_tensor
        box_wh /= wh_tensor
        # box_xy -= (box_wh / 2.)  # from center to conner
        boxes = torch.cat((box_xy, box_wh), -1) * image_tensor  # to origin image scale
        boxes = boxes.view(b, -1, 4)

        output = torch.cat((boxes, box_conf, box_classes.float()), -1)
        # return boxes, box_conf, box_classes
        return output


def main():
    model = CRNet()
    model_dict = model.state_dict()
    print(model_dict.keys())
    weights_count = model.count_weights()
    print('Weight counted: ', weights_count)

    model.load_weights('./data/ocr_new/lp-recognition.weights')
    torch.save(model, 'crnet_pytorch_new.pt')

    model.to('cuda')
    img = torch.ones(1, 3, 80, 240).cuda()
    with torch.no_grad():
        for i in range(25):
            tic = time.time()
            out = model(img)
            torch.cuda.synchronize()

            print('FPS: ', 1 / (time.time() - tic))


if __name__ == '__main__':
    main()
