import cv2
import numpy as np
from message import Message
from data_loader import DataLoader
from grouping import Grouping
from group_shape_generation import GroupShapeGeneration

dataset = 'ucy'
dataset_idx = 1
history = 16
frame_idx = 500
group_idx = 20

msg = Message()
data = DataLoader(dataset, dataset_idx)
msg = data.update_message(msg)
gp = Grouping(msg, 16)
msg = gp.update_message(msg)
print(msg.video_labels_matrix[frame_idx])

gs_gen = GroupShapeGeneration(msg)
vertices, pedidx = gs_gen.generate_group_shape(frame_idx, group_idx, None)
print(vertices)
print(pedidx)

canvas = np.zeros((msg.frame_height, msg.frame_width, 3), dtype=np.uint8)
canvas, pedidx = gs_gen.generate_group_shape(frame_idx, group_idx, canvas)
cv2.imwrite('example.jpg', canvas)
