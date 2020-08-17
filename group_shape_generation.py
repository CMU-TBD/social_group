import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import norm
from scipy.stats import truncnorm

class GroupShapeGeneration(object):

    def __init__(self, msg):
        if msg.if_processed_data:
            self.H = msg.H
            self.dataset = msg.dataset
            self.frame_width = msg.frame_width
            self.frame_height = msg.frame_height
            self.video_position_matrix = msg.video_position_matrix
            self.video_velocity_matrix = msg.video_velocity_matrix
            self.video_pedidx_matrix = msg.video_pedidx_matrix
        else:
            raise Exception('Data has not been loaded!')
        if msg.if_processed_group:
            self.video_labels_matrix = msg.video_labels_matrix
        else:
            raise Exception('Grouping has not been performed!')
        return

    def _find_label_properties(self, frame_idx, label):
        positions = []
        velocities = []
        pedidx = []
        labels = self.video_labels_matrix[frame_idx]
        for i in range(len(labels)):
            if label == labels[i]:
                positions.append(self.video_position_matrix[frame_idx][i])
                velocities.append(self.video_velocity_matrix[frame_idx][i])
                pedidx.append(self.video_pedidx_matrix[frame_idx][i])
        return positions, velocities, pedidx

    def _coordinate_transform(self, coord):
        pt = np.matmul(np.linalg.inv(self.H), [[coord[0]], [coord[1]], [1.0]])
        x = pt[0][0] / pt[2][0]
        y = pt[1][0] / pt[2][0]
        if self.dataset == 'ucy':
            tmp_y = y
            y = self.frame_width / 2 + x
            x = self.frame_height / 2 - tmp_y
        x = int(round(x))
        y = int(round(y))
        return x, y

    def _draw_social_shapes(self, position, velocity, frame):
        front_coeff = 1.0
        side_coeff = 2.0 / 3.0
        rear_coeff = 0.5
        total_increments = 20
        quater_increments = total_increments / 4
        angle_increment = 2 * np.pi / total_increments
        current_target = 0.8

        contour_points = []
        for i in range(len(position)):
            center_x = position[i][0]
            center_y = position[i][1]
            velocity_x = velocity[i][0]
            velocity_y = velocity[i][1]

            velocity_magnitude = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
            velocity_angle = np.arctan2(velocity_y, velocity_x)
            variance_front = max(0.5, front_coeff * velocity_magnitude)
            variance_side = side_coeff * variance_front
            variance_rear = rear_coeff * variance_front

            for j in range(total_increments):
                if (j / quater_increments) == 0:
                    prev_variance = variance_front
                    next_variance = variance_side
                elif (j / quater_increments) == 1:
                    prev_variance = variance_rear
                    next_variance = variance_side
                elif (j / quater_increments) == 2:
                    prev_variance = variance_rear
                    next_variance = variance_side
                else:
                    prev_variance = variance_front
                    next_variance = variance_side
                current_variance = prev_variance + (next_variance - prev_variance) * \
                                   (j % quater_increments) / float(quater_increments)
                value = np.sqrt(0.354163 / ((np.cos(angle_increment * j) ** 2 / (2 * prev_variance)) + (np.sin(angle_increment * j) ** 2 / (2 * next_variance))))

                addition_angle = velocity_angle + angle_increment * j
                x = center_x + np.cos(addition_angle) * value
                y = center_y + np.sin(addition_angle) * value
                contour_points.append((x, y))

        convex_hull_vertices = []
        hull = ConvexHull(np.array(contour_points))
        for i in hull.vertices:
            hull_vertice = (contour_points[i][0], contour_points[i][1])
            convex_hull_vertices.append(hull_vertice)

        if frame is None:
            return convex_hull_vertices
        else:
            for i, elem in enumerate(convex_hull_vertices):
                x, y = self._coordinate_transform(elem)
                convex_hull_vertices[i] = (y, x)
            cv2.fillConvexPoly(frame, np.array(convex_hull_vertices), (255, 255, 255))
            return frame

    def generate_group_shape(self, frame_idx, group_label, frame=None):
        positions, velocities, pedidx = self._find_label_properties(frame_idx, group_label)
        if len(positions) == 0:
            raise Exception('Group does not exist in the given frame!')
        if frame is None:
            vertices = self._draw_social_shapes(positions, velocities, None)
            return vertices, pedidx
        else:
            frame = self._draw_social_shapes(positions, velocities, frame)
            return frame, pedidx
