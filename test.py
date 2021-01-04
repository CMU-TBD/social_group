import numpy as np
import cv2
from message import Message
from data_loader import DataLoader
from group_shape_prediction import GroupShapePrediction

group_center_1 = [-5, -5]
group_center_2 = [5, 5]


ped1 = []
ped2 = []
ped3 = []
ped4 = []
ped5 = []
ped6 = []
ped7 = []
ped8 = []
for i in range(8):
    ped1.append([group_center_1[0], group_center_1[1] + i * 0.32])
    ped2.append([group_center_1[0] + 1, group_center_1[1] + i * 0.32])
    ped3.append([group_center_1[0], group_center_1[1] + i * 0.32 + 1])
    ped4.append([group_center_1[0] + 1, group_center_1[1] + i * 0.32 + 1])

    ped5.append([group_center_2[0], group_center_2[1] - i * 0.32])
    ped6.append([group_center_2[0] + 1, group_center_2[1] - i * 0.32])
    ped7.append([group_center_2[0], group_center_2[1] - i * 0.32 + 1])
    ped8.append([group_center_2[0] + 1, group_center_2[1] - i * 0.32 + 1])

ped = [ped1, ped2, ped3, ped4, ped5, ped6, ped7, ped8]
vel = [[[0, 4]] * 8] * 4 + [[[0, -4]] * 8] * 4

dataset = 'eth'
dataset_idx = 0
msg = Message()
data = DataLoader(dataset, dataset_idx)
msg = data.update_message(msg)

gsp = GroupShapePrediction(msg)
output_seq = gsp.predict(ped, vel)

for i, output in enumerate(output_seq):
    img = np.repeat(np.expand_dims(output * 255, axis=2), 3, axis=2)
    cv2.imwrite('tmp/' + str(i) + '.jpg', img)
