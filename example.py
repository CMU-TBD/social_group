import cv2
import numpy as np
from message import Message
from data_loader import DataLoader
from grouping import Grouping
from group_shape_generation import GroupShapeGeneration
from img_process import DrawGroupShape

dataset = 'ucy'
dataset_idx = 0
history = 16
frame_idx = 500
group_idx = 20

# Initialize a message
msg = Message()

# Initialize dataloader
data = DataLoader(dataset, dataset_idx)

# Update the message
msg = data.update_message(msg)

# Initialize grouping
gp = Grouping(msg, 16)

# Update the message
msg = gp.update_message(msg)
# This shows what group ids are in this frame
print(msg.video_labels_matrix[frame_idx])

# Initialize group shape generation
gs_gen = GroupShapeGeneration(msg)
vertices, pedidx = gs_gen.generate_group_shape(frame_idx, group_idx)
# The returned vertices for the group shape
print(vertices)
print(pedidx)

# We can also draw it on an image (blank canvas in this case)
canvas = np.zeros((msg.frame_height, msg.frame_width, 3), dtype=np.uint8)
dgs = DrawGroupShape(msg)
canvas = dgs.draw_group_shape(vertices, canvas)
cv2.imwrite('example.jpg', canvas)
