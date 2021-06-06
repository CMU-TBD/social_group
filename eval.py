import numpy as np
import torch
import cv2
from data_generation import DataGeneration as dg
from model import ConvAutoencoder as ca

def baseline(input_sequence, pred_length):
    input_sequence = input_sequence[:, :, :, 0]
    input_last = input_sequence[-1, :, :]
    input_2nd_last = input_sequence[-2, :, :]
    
    coords_last = np.nonzero(input_last)
    coords_2nd_last = np.nonzero(input_2nd_last)
    geo_center_last = np.mean(coords_last, axis=1)
    geo_center_2nd_last = np.mean(coords_2nd_last, axis=1)
    spd = geo_center_last - geo_center_2nd_last

    output_sequence = np.zeros((pred_length, 224, 224, 1))
    for i in range(pred_length):
        frame_coords = np.array([coords_last[0] + spd[0] * (i + 1), 
                                 coords_last[1] + spd[1] * (i + 1)])
        for j in range(np.shape(frame_coords)[1]):
            x = int(frame_coords[0, j])
            y = int(frame_coords[1, j])
            if (x >= 0) and (y >= 0) and (x < 224) and (y < 224):
                output_sequence[i, x, y, 0] = 1

    return output_sequence

def get_IoU(output_gt, output):
    num_gt = len(np.nonzero(output_gt)[0])
    num_inf = len(np.nonzero(output)[0])
    num_U = len(np.nonzero(output + output_gt)[0])
    num_I = num_gt + num_inf - num_U
    IoU = num_I / num_U
    return IoU

def inf_mode_0(input_sequence, pred_length, autoencoder, cuda):
    output_sequence = []
    for i in range(pred_length):
        inputs = np.transpose(np.array(input_sequence), (3, 0, 1, 2))
        inputs_tensor = np.expand_dims(inputs, 0)
        inputs_tensor = torch.tensor(inputs_tensor, dtype=torch.float32, device=cuda)
     
        outputs_tensor = autoencoder(inputs_tensor)
        outputs = outputs_tensor.data.cpu().numpy()
        outputs = np.transpose(outputs[0, :, :, :], (1, 2, 0))
        output_sequence.append(outputs)
     
        input_sequence = input_sequence[1:]
        input_sequence.append(outputs)
    return output_sequence

def inf_mode_1(input_sequence, autoencoder, cuda):
    inputs = np.transpose(np.array(input_sequence), (3, 0, 1, 2))
    inputs_tensor = np.expand_dims(inputs, 0)
    inputs_tensor = torch.tensor(inputs_tensor, dtype=torch.float32, device=cuda)
    outputs_tensor = autoencoder(inputs_tensor)
    outputs = outputs_tensor.data.cpu().numpy()
    output_sequence = np.transpose(outputs[0, :, :, :], (1, 2, 3, 0))
    return output_sequence

mode = 1

test_num = 2
cuda = torch.device('cuda:2')
if mode == 0:
    ckpt = 'checkpoints/model_fpsfix_' + str(test_num) + '_200.pth'
else:
    ckpt = 'checkpoints/model_fpsfix_' + str(test_num) + '_200.pth'
print('Loading: {}'.format(ckpt))
autoencoder = ca()
autoencoder.load_state_dict(torch.load(ckpt, map_location='cpu'))
autoencoder.eval()
autoencoder.to(cuda)

history = 8
pred_length = 8
offset = 2
data_generator = dg(history, offset, [], [test_num])

threshold = 0.05

######################################################################################
inputs_raw, outputs_raw = data_generator.generate_sample(from_train=False)
for i in range(history):
    img = inputs_raw[i, :, :, :]
    cv2.imwrite('test/' + str(i) + '.jpg', np.uint8(img * 255))

input_sequence = list(inputs_raw)
if mode == 0:
    output_sequence = inf_mode_0(input_sequence, pred_length, autoencoder, cuda)
else:
    output_sequence = inf_mode_1(input_sequence, autoencoder, cuda)

for i in range(pred_length):
    output_gt = outputs_raw[i]
    output_inf = output_sequence[i]
    cv2.imwrite('test/' + str(i) + '_gt.jpg', np.uint8(output_gt * 255))
    cv2.imwrite('test/' + str(i) + '_inf.jpg', np.uint8(output_inf * 255))
######################################################################################


######################################################################################
input_cases, output_cases = data_generator.generate_cases_all_groups(test_num)
num_cases = len(input_cases)
avg_IoU = 0
avg_final_IoU = 0
avg_bs_IoU = 0
avg_final_bs_IoU = 0
for i in range(num_cases):
    input_sequence = list(input_cases[i])
    output_gt_sequence = list(output_cases[i])
    if mode == 0:
        output_inf_sequence = inf_mode_0(input_sequence, pred_length, autoencoder, cuda)
    else:
        output_inf_sequence = inf_mode_1(input_sequence, autoencoder, cuda)
    output_bs_sequence = baseline(input_cases[i], pred_length)

    sample_IoU = 0
    sample_bs_IoU = 0
    for j in range(pred_length):
        output_gt = output_gt_sequence[j] >= threshold
        output_inf = output_inf_sequence[j] >= threshold
        output_bs = output_bs_sequence[j] >= threshold
        
        IoU = get_IoU(output_gt, output_inf)
        IoU_bs = get_IoU(output_gt, output_bs)
        sample_IoU += IoU
        sample_bs_IoU += IoU_bs
        if j == (pred_length - 1):
            final_IoU = IoU
            final_bs_IoU = IoU_bs

    print([i, num_cases], end='\r')
    avg_IoU += sample_IoU / pred_length
    avg_final_IoU += final_IoU
    avg_bs_IoU += sample_bs_IoU / pred_length
    avg_final_bs_IoU += final_bs_IoU

print('The average baseline IoU of all cases is: {}'.format(avg_bs_IoU / num_cases))
print('The average baseline final IoU of all cases is: {}'.format(avg_final_bs_IoU / num_cases))
print('The average IoU of all cases is: {}'.format(avg_IoU / num_cases))
print('The average final IoU of all cases is: {}'.format(avg_final_IoU / num_cases))
######################################################################################


"""
######################################################################################
num_epoch = 201
interval = 10
for e in range(0, num_epoch, interval):
    if mode == 0:
        ckpt = 'checkpoints/model_fpsfix_' + str(test_num) + '_' + str(e) + '.pth'
    else:
        ckpt = 'checkpoints/model_fpsfix_' + str(test_num) + '_' + str(e) + '.pth'
    autoencoder = ca()
    autoencoder.load_state_dict(torch.load(ckpt, map_location='cpu'))
    autoencoder.eval()
    autoencoder.to(cuda)

    avg_IoU = 0
    avg_final_IoU = 0
    for i in range(num_cases):
        input_sequence = list(input_cases[i])
        output_gt_sequence = list(output_cases[i])
        if mode == 0:
            output_inf_sequence = inf_mode_0(input_sequence, pred_length, autoencoder, cuda)
        else:
            output_inf_sequence = inf_mode_1(input_sequence, autoencoder, cuda)

        sample_IoU = 0
        for j in range(pred_length):
            output_gt = np.round(output_gt_sequence[j])
            output_inf = np.round(output_inf_sequence[j])
            
            IoU = get_IoU(output_gt, output_inf)
            sample_IoU += IoU
            if j == (pred_length - 1):
                final_IoU = IoU

        avg_IoU += sample_IoU / pred_length
        avg_final_IoU += final_IoU

    avg_IoU /= num_cases
    avg_final_IoU /= num_cases
    print([e, avg_IoU, avg_final_IoU])
######################################################################################
"""
