import cv2
import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import norm
from scipy.stats import truncnorm

from matplotlib import pyplot as plt

class GroupShapeGeneration(object):

    # This class takes the information from grouping and
    # generates social group shapes.
    # Group shapes are generated by calling the generate_group_shape
    # method after classs initialization.
    #
    # Group shapes are formatted as either a sequence of coordinates
    # in the counter clock direction representing the convex shape blob,
    # or, if an image is given, formatted as drawing the blob on the image.

    def __init__(self, msg):
        # Initialization
        # Inputs:
        # msg: message class object (should have data loaded first)

        if msg.if_processed_data:
            self.video_position_matrix = msg.video_position_matrix
            self.video_velocity_matrix = msg.video_velocity_matrix
            self.video_pedidx_matrix = msg.video_pedidx_matrix
        else:
            raise Exception('Data has not been loaded!')
        if msg.if_processed_group:
            self.video_labels_matrix = msg.video_labels_matrix
        else:
            raise Exception('Grouping has not been performed!')

        if (msg.dataset == 'ucy') and (msg.flag == 2):
            self.const = 0.25
        else:
            self.const = 0.35
        return

    def _find_label_properties(self, frame_idx, label):
        # Given a frame index and a group membership label (group id),
        # find information (positions, velocities, person ids) of
        # all the pedestrians in that group and that frame.
        # Inputs:
        # frame_idx: frame index of the video
        # label: group membership label
        # Outputs:
        # positions: positions of the pedestrians in the given group and frame index.
        # velocities: velocities of the pedestrians in the given group and frame index.
        # pedidx: person ids of the pedestrians in the given group and frame index.

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
    
    @staticmethod
    def draw_social_shapes(position, velocity, const):
        # This function draws social group shapes
        # given the positions and velocities of the pedestrians.

        # Parameters from Rachel Kirby's thesis
        front_coeff = 1.0
        side_coeff = 2.0 / 3.0
        rear_coeff = 0.5
        safety_dist = 0.5
        total_increments = 20 # controls the resolution of the blobs
        quater_increments = total_increments / 4
        angle_increment = 2 * np.pi / total_increments

        # Draw a personal space for each pedestrian within the group
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

            # Draw four quater-ovals with the axis determined by front, side and rear "variances"
            # The overall shape contour does not have discontinuities.
            for j in range(total_increments):
                if (j // quater_increments) == 0:
                    prev_variance = variance_front
                    next_variance = variance_side
                elif (j // quater_increments) == 1:
                    prev_variance = variance_rear
                    next_variance = variance_side
                elif (j // quater_increments) == 2:
                    prev_variance = variance_rear
                    next_variance = variance_side
                else:
                    prev_variance = variance_front
                    next_variance = variance_side
                value = np.sqrt(const / ((np.cos(angle_increment * j) ** 2 / (2 * prev_variance)) + (np.sin(angle_increment * j) ** 2 / (2 * next_variance))))
                value = max(safety_dist, value)
                #value = 0.5

                addition_angle = velocity_angle + angle_increment * j
                x = center_x + np.cos(addition_angle) * value
                y = center_y + np.sin(addition_angle) * value
                contour_points.append((x, y))

        #plt.scatter(np.array(contour_points)[:, 0], np.array(contour_points)[:, 1])
        #plt.gca().set_aspect('equal', adjustable='box')
        #plt.draw()
        #plt.show()

        # Get the convex hull of all the personal spaces
        convex_hull_vertices = []
        hull = ConvexHull(np.array(contour_points))
        for i in hull.vertices:
            hull_vertice = (contour_points[i][0], contour_points[i][1])
            convex_hull_vertices.append(hull_vertice)

        return convex_hull_vertices

    def generate_group_shape(self, frame_idx, group_label):
        # Method that generates group shape
        # Inputs
        # frame_idx: frame number
        # group_label: group id
        # frame(optional): an image in numpy array (opencv image format)
        # Outputs
        # If an image frame is not provided
        # vertices: coordinates (in meters) that draws a convex shape blob 
        #           in the clockwise direction.
        # If an image frame is provided
        # frame: an updated image with a group shape drawn on it.
        # (Regardless) pedidx: the person ids of the pedestrians in the given group.

        positions, velocities, pedidx = self._find_label_properties(frame_idx, group_label)
        if len(positions) == 0:
            raise Exception('Group does not exist in the given frame!')
        vertices = self.draw_social_shapes(positions, velocities, self.const)
        return vertices, (positions, velocities, pedidx)
