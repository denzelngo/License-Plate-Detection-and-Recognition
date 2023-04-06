from keras.models import model_from_json
import numpy as np
import torch
from wpod_pytorch import WPODNet

with open('data/lp-detector/wpod-net_update1.json', 'r') as json_file:
	model_json = json_file.read()


model_keras = model_from_json(model_json)
model_keras.load_weights('data/lp-detector/wpod-net_update1.h5')

model_pytorch = WPODNet()
pytorch_weights_dict = dict()


# Load weights to pytorch state dict from keras model weights
res_ind_conv = 0
res_ind_bn = 0
for layer in model_keras.layers:
	layer_name = layer.get_config()['name']
	if layer_name == 'input': continue
	ind = layer_name.split('_')[-1]
	if not int(ind) in [4, 5, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]:
		if 'conv' in layer_name:
			pytorch_weights_dict['conv_' + ind + '.conv2d.weight'] = torch.from_numpy(
				np.transpose(layer.get_weights()[0], (3, 2, 0, 1)))
			pytorch_weights_dict['conv_' + ind + '.conv2d.bias'] = torch.from_numpy(layer.get_weights()[1])
		# print('{} loaded.'.format('conv_' + ind + '.conv2d'))
		if 'batch' in layer_name:
			pytorch_weights_dict['conv_' + ind + '.bn.weight'] = torch.from_numpy(layer.get_weights()[0])
			pytorch_weights_dict['conv_' + ind + '.bn.bias'] = torch.from_numpy(layer.get_weights()[1])
			pytorch_weights_dict['conv_' + ind + '.bn.running_mean'] = torch.from_numpy(layer.get_weights()[2])
			pytorch_weights_dict['conv_' + ind + '.bn.running_var'] = torch.from_numpy(layer.get_weights()[3])
	# print('{} loaded.'.format('conv_' + ind + '.bn'))
	elif int(ind) in [25, 26]:
		pytorch_weights_dict['conv_' + ind + '.weight'] = torch.from_numpy(
			np.transpose(layer.get_weights()[0], (3, 2, 0, 1)))
		pytorch_weights_dict['conv_' + ind + '.bias'] = torch.from_numpy(layer.get_weights()[1])
	elif int(ind) in [4, 7, 9, 12, 14, 17, 19, 21, 23]:
		if 'conv' in layer_name:
			res_ind_conv += 1
			pytorch_weights_dict['res_block_' + str(res_ind_conv) + '.conv2d_1.weight'] = torch.from_numpy(
				np.transpose(layer.get_weights()[0], (3, 2, 0, 1)))
			pytorch_weights_dict['res_block_' + str(res_ind_conv) + '.conv2d_1.bias'] = torch.from_numpy(
				layer.get_weights()[1])
		if 'batch' in layer_name:
			res_ind_bn += 1
			pytorch_weights_dict['res_block_' + str(res_ind_bn) + '.bn_1.weight'] = torch.from_numpy(
				layer.get_weights()[0])
			pytorch_weights_dict['res_block_' + str(res_ind_bn) + '.bn_1.bias'] = torch.from_numpy(
				layer.get_weights()[1])
			pytorch_weights_dict['res_block_' + str(res_ind_bn) + '.bn_1.running_mean'] = torch.from_numpy(
				layer.get_weights()[2])
			pytorch_weights_dict['res_block_' + str(res_ind_bn) + '.bn_1.running_var'] = torch.from_numpy(
				layer.get_weights()[3])
	else:
		if 'conv' in layer_name:
			pytorch_weights_dict['res_block_' + str(res_ind_conv) + '.conv2d_2.weight'] = torch.from_numpy(
				np.transpose(layer.get_weights()[0], (3, 2, 0, 1)))
			pytorch_weights_dict['res_block_' + str(res_ind_conv) + '.conv2d_2.bias'] = torch.from_numpy(
				layer.get_weights()[1])
		if 'batch' in layer_name:
			pytorch_weights_dict['res_block_' + str(res_ind_bn) + '.bn_2.weight'] = torch.from_numpy(
				layer.get_weights()[0])
			pytorch_weights_dict['res_block_' + str(res_ind_bn) + '.bn_2.bias'] = torch.from_numpy(
				layer.get_weights()[1])
			pytorch_weights_dict['res_block_' + str(res_ind_bn) + '.bn_2.running_mean'] = torch.from_numpy(
				layer.get_weights()[2])
			pytorch_weights_dict['res_block_' + str(res_ind_bn) + '.bn_2.running_var'] = torch.from_numpy(
				layer.get_weights()[3])

model_pytorch.load_state_dict(pytorch_weights_dict)
torch.save(model_pytorch, 'lp_detector.pt')

print('Done!')
