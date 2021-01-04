import cv2
import numpy as np
from message import Message
from data_loader import DataLoader
from grouping import Grouping
from group_shape_generation import GroupShapeGeneration
from img_process import ProcessImage, DrawGroupShape

class DataGeneration(object):

    def __init__(self, history, offset, train_set, test_set):
        self.history = history
        self.history += 1
        self.offset = offset
        self.history *= offset
        self.msg_group = self._get_msg_group()
        self.train_idx = train_set
        self.test_idx = test_set

        self.train_prob = self._calculate_set_prob(self.train_idx)
        self.test_prob = self._calculate_set_prob(self.test_idx)
        self.group_labels, self.frame_labels = self._get_labels()

        self.num_train_data = 0
        self.num_test_data = 0
        for e in self.train_idx:
            self.num_train_data += len(self.group_labels[e])
        for e in self.test_idx:
            self.num_test_data += len(self.group_labels[e])
        return

    def _get_msg_group(self):
        msg_group = []
        dataset_list = ['eth', 'eth', 'ucy', 'ucy', 'ucy']
        dataset_idx_list = [0, 1, 0, 1, 2]
        #dataset_list = ['eth']
        #dataset_idx_list = [0]

        for i in range(len(dataset_list)):
            dataset = dataset_list[i]
            dataset_idx = dataset_idx_list[i]
            msg = Message()
            data = DataLoader(dataset, dataset_idx)
            msg = data.update_message(msg)
            gp = Grouping(msg, self.history)
            msg = gp.update_message(msg)
            msg_group.append(msg)

        return msg_group

    def _calculate_set_prob(self, set_idx):
        set_prob = []
        for i in set_idx:
            msg = self.msg_group[i]
            gp_labels = msg.video_labels_matrix
            set_prob.append(len(self._get_unique_labels(gp_labels)))
        set_prob = np.array(set_prob)
        set_prob = set_prob / np.sum(set_prob)
        return set_prob

    def _get_labels(self):
        group_labels = []
        frame_labels = []
        for msg in self.msg_group:
            msg_group_labels = []
            msg_frame_labels = []
            tmp_frame_labels = []
            labels = msg.video_labels_matrix
            max_label = np.max(self._get_unique_labels(labels))
            for i in range(max_label + 1):
                tmp_frame_labels.append([])
            for i, sub_list in enumerate(labels):
                for elem in sub_list:
                    tmp_frame_labels[elem].append(i)
            
            for i, sub_list in enumerate(tmp_frame_labels):
                sub_list = np.unique(sub_list)
                if not(len(sub_list) < self.history):
                    msg_group_labels.append(i)
                    msg_frame_labels.append(sub_list[(self.history - 1):None])
            group_labels.append(msg_group_labels)
            frame_labels.append(msg_frame_labels)
        return group_labels, frame_labels
   
    def _get_unique_labels(self, labels):
        all_labels = []
        for sub_list in labels:
            all_labels += sub_list
        return np.unique(all_labels)

    def generate_sample(self, from_train=True, debug=False):
        if from_train:
            idx = np.random.choice(self.train_idx, p=self.train_prob)
        else:
            idx = np.random.choice(self.test_idx, p=self.test_prob)
        msg = self.msg_group[idx]
        shape_gen_class = GroupShapeGeneration(msg)

        group_pool = self.group_labels[idx]
        #print(len(group_pool))
        frame_pool = self.frame_labels[idx]
        if (len(group_pool) == 0):
            raise Exception('No valid groups exist!')
        group_idx = np.random.choice(range(len(group_pool)))
        group = group_pool[group_idx]
        frame = np.random.choice(frame_pool[group_idx])
        img_seq = self._generate_img_sequence(shape_gen_class, msg, group, frame, debug, from_train)
        return np.array(img_seq[:-1]), np.array(img_seq[-1])

    def _generate_img_sequence(self, shape_gen_class, msg, group, frame, debug, from_train):
        vertice_sequence = []
        for i in range(frame - self.history + 1, frame + 1, self.offset):
            vertices, _ = shape_gen_class.generate_group_shape(i, group)
            vertice_sequence.append(vertices)

        dgs = DrawGroupShape(msg)
        dgs.set_center(vertice_sequence[:-1])
        dgs.set_aug()
        img_sequence = []
        for i, v in enumerate(vertice_sequence):
            canvas = np.zeros((msg.frame_height, msg.frame_width, 3), dtype=np.uint8)
            img = dgs.draw_group_shape(v, canvas, center=True, aug=from_train)
            img_sequence.append(img)

        pimg = ProcessImage(msg, img_sequence[:-1])
        for i, img in enumerate(img_sequence):
            img_sequence[i] = pimg.process_image(img, debug)

        return img_sequence

