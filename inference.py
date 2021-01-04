import numpy as np
import torch
import cv2
from data_generation import DataGeneration as dg
from model import ConvAutoencoder as ca

cuda = torch.device('cuda:1')
ckpt = 'checkpoints/model_4_200.pth'
autoencoder = ca()
autoencoder.load_state_dict(torch.load(ckpt))
autoencoder.eval()
autoencoder.to(cuda)

history = 16
offset = 3
data_generator = dg(history, offset)

inputs, outputs = data_generator.generate_sample(from_train=False)
for i in range(history):
    img = inputs[i, :, :, :]
    cv2.imwrite('test/' + str(i) + '.jpg', np.uint8(img * 255))

inputs = np.transpose(inputs, (3, 0, 1, 2))
outputs = np.transpose(outputs, (2, 0, 1))
inputs_tensor = np.expand_dims(inputs, 0)
outputs_tensor = np.expand_dims(outputs, 0)
inputs_tensor = torch.tensor(inputs_tensor, dtype=torch.float32, device=cuda)
outputs_tensor = torch.tensor(outputs_tensor, dtype=torch.float32, device=cuda)

outputs_model = autoencoder(inputs_tensor)

output_gt = outputs_tensor.data.cpu().numpy()
output_inf = outputs_model.data.cpu().numpy()
output_gt = np.repeat(np.transpose(output_gt[0, :, :, :], (1, 2, 0)), 3, axis=2)
output_inf = np.repeat(np.transpose(output_inf[0, :, :, :], (1, 2, 0)), 3, axis=2)
cv2.imwrite('test/gt.jpg', np.uint8(output_gt * 255))
cv2.imwrite('test/inf.jpg', np.uint8(output_inf * 255))
