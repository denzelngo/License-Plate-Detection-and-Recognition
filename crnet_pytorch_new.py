import torch
import numpy as np
from torch import nn
from cr_net_utils.anchorBox import AnchorBox_new
from cr_net_utils.detect import Detect_new
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
	def __init__(self):
		super(CRNet, self).__init__()
		self.anchor_box = AnchorBox_new().forward()

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

		print('{}/{} weights loaded.'.format(ptr, len(weights)))

	def forward(self, x, img_shape=(352, 128), batch=True):
		out = self.darknet(x)
		if batch:
			return out
		else:
			b, c, h, w = out.size()  # [N,200,16,44]
			feat = out.permute(0, 2, 3, 1).contiguous().view(b, -1, 5, 35 + 5)
			# feat = out.view(b, 2, 35 + 5, -1).permute(0, 1, 3, 2).contiguous()
			box_xy, box_wh = torch.sigmoid(feat[..., 0:2]), feat[..., 2:4].exp()
			box_conf, score_pred = torch.sigmoid(feat[..., 4:5]), feat[..., 5:].contiguous()

			box_prob = torch.softmax(score_pred, dim=3)
			box_pred = torch.cat([box_xy, box_wh], 3)
			# print('Score: ', box_pred)
			width, height = img_shape
			img_shape = torch.Tensor([[width, height, width, height]])
			self.anchor_box, img_shape = self.anchor_box.cuda(), img_shape.cuda()
			self.anchor_box = self.anchor_box.view_as(box_pred)
			return Detect_new()(box_pred, box_conf, box_prob, self.anchor_box, img_shape)


def main():
	model = CRNet()
	model_dict = model.state_dict()
	print(model_dict.keys())
	weights_count = model.count_weights()
	print('Weight counted: ', weights_count)

	model.load_weights('./data/ocr_new/lp-recognition.weights')
	torch.save(model,'crnet_pytorch_new.pt')

	model.cuda()
	img = torch.ones(1, 3, 128, 352).cuda()
	with torch.no_grad():
		for i in range(25):
			tic = time.time()
			out = model(img)

			print('FPS: ', 1/(time.time()-tic))


if __name__ == '__main__':
	main()
