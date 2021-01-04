import numpy as np
import torch
import cv2
from grouping import Grouping
from group_shape_generation import GroupShapeGeneration
from img_process import ProcessImage, DrawGroupShape
from model import ConvAutoencoder

class GroupShapePrediction(object):

    def __init__(self, msg):
        # No need to do grouping here for msg
        self.msg = msg
        return

    def _load_parameters(self):
        # Initialize parameters to prepare for DBSCAN

        pos = 2.0
        ori = 30
        vel = 1.0
        params = {'position_threshold': pos,
                  'orientation_threshold': ori / 180.0 * np.pi,
                  'velocity_threshold': vel,
                  'velocity_ignore_threshold': 0.5}
        return params

    def _predict_sequence(self, input_sequence, pred_length):
        cuda = torch.device('cuda:0')
        ckpt = 'checkpoints/model_0_200.pth'
        model = ConvAutoencoder()
        model.load_state_dict(torch.load(ckpt))
        model.eval()
        model.to(cuda)

        output_sequence = []
        for i in range(pred_length):
            inputs = np.transpose(np.array(input_sequence), (3, 0, 1, 2))
            inputs_tensor = np.expand_dims(inputs, 0)
            inputs_tensor = torch.tensor(inputs_tensor, dtype=torch.float32, device=cuda)
            
            outputs_tensor = model(inputs_tensor)
            outputs = outputs_tensor.data.cpu().numpy()
            outputs = np.transpose(outputs[0, :, :, :], (1, 2, 0))
            output_sequence.append(outputs)
            
            input_sequence = input_sequence[1:]
            input_sequence.append(outputs)

        return output_sequence

    def predict(self, positions, velocities):
        params = self._load_parameters()
        
        position_array = []
        velocity_array = []
        num_people = len(positions)

        if num_people == 0:
            raise Exception('People Needed!')

        seq_length = len(positions[0])
        pred_seq_length = 8
        #gp = Grouping(self.msg, seq_length)
        #self.msg = gp.update_message(self.msg)
        for i in range(num_people):
            position_array.append(positions[i][-1])
            velocity_array.append(velocities[i][-1])    
            labels = Grouping.grouping(position_array, velocity_array, params)

        all_labels = np.unique(labels)
        num_groups = len(all_labels)
        all_pred_img_sequences = []
        #gsg = GroupShapeGeneration(self.msg)
        for curr_label in all_labels:
            group_positions = []
            group_velocities = []
            for i, l in enumerate(labels):
                if l == curr_label:
                    group_positions.append(positions[i])
                    group_velocities.append(velocities[i])
            
            vertice_sequence = []
            for i in range(seq_length):
                frame_positions = []
                frame_velocities = []
                for j in range(len(group_positions)):
                    frame_positions.append(group_positions[j][i])
                    frame_velocities.append(group_velocities[j][i])
                vertices = GroupShapeGeneration.draw_social_shapes(frame_positions, 
                                                                   frame_velocities)
                vertice_sequence.append(vertices)

            dgs = DrawGroupShape(self.msg)
            dgs.set_center(vertice_sequence)
            dgs.set_aug(angle=0)
            img_sequence = []
            for i, v in enumerate(vertice_sequence):
                canvas = np.zeros((self.msg.frame_height, self.msg.frame_width, 3), dtype=np.uint8)
                img = dgs.draw_group_shape(v, canvas, center=True, aug=False)
                img_sequence.append(img)

            pimg = ProcessImage(self.msg, img_sequence)
            for i, img in enumerate(img_sequence):
                img_sequence[i] = pimg.process_image(img, debug=False) 

            pred_img_sequence = self._predict_sequence(img_sequence, pred_seq_length)

            group_pred_img_sequence = []
            for i, img in enumerate(pred_img_sequence):
                #img = np.round(np.repeat(img, 3, axis=2)) * 255
                img = np.round(np.repeat(img, 3, axis=2))
                pred_img = pimg.reverse_process_image(img, debug=True)
                pred_img = dgs.reverse_move_center_img(pred_img)
                group_pred_img_sequence.append(pred_img[:, :, 0])
            all_pred_img_sequences.append(group_pred_img_sequence)

        fnl_pred_img_sequence = []
        for i in range(pred_seq_length):
            canvas = np.zeros((self.msg.frame_height, self.msg.frame_width), dtype=np.uint8)
            for j in range(num_groups):
                img = all_pred_img_sequences[j][i]
                #img = np.round(img)
                canvas += img
            fnl_pred_img_sequence.append(np.clip(canvas, 0, 1))

        return fnl_pred_img_sequence
