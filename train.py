from data_generation import DataGeneration as dg
from model import ConvAutoencoder as ca
import torch
import numpy as np
import cv2
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

history = 8
offset = 2
train_set = [1,2,3,4]
test_set = [0]
data_generator = dg(history, offset, train_set, test_set)

cuda = torch.device('cuda:0')
autoencoder = ca()
autoencoder = autoencoder.to(cuda)


batch_size = 1
num_epoch = 200
num_data = data_generator.num_train_data
print(num_data)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.00001)
for epoch in range(num_epoch + 1):
    train_loss = 0.0
    for i in range(num_data):
        inputs_tensor = []
        outputs_tensor = []
        for b in range(batch_size):
            inputs, outputs = data_generator.generate_sample()
            inputs = np.transpose(inputs, (3, 0, 1, 2))
            outputs = np.transpose(outputs, (2, 0, 1))
            inputs_tensor.append(inputs)
            outputs_tensor.append(outputs)
        inputs_tensor = torch.tensor(np.array(inputs_tensor), dtype=torch.float32, device=cuda)
        outputs_tensor = torch.tensor(np.array(outputs_tensor), dtype=torch.float32, device=cuda)

        optimizer.zero_grad()
        outputs_model = autoencoder(inputs_tensor)

        loss = criterion(outputs_model, outputs_tensor)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_size

    train_loss = train_loss / num_data
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

    if epoch % 5 == 0:
        torch.save(autoencoder.state_dict(), 'checkpoints/model_0_{}.pth'.format(epoch))

