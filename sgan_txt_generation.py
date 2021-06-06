import numpy as np

from data_loader import DataLoader as DL

fps = 10

dsets = ['eth', 'eth', 'ucy', 'ucy', 'ucy']
dnum = [0, 1, 0, 1, 2]

for idx in range(len(dsets)):
    ds = DL(dsets[idx], dnum[idx], fps)
    num_frames = len(ds.video_position_matrix)
    with open('sgan_dataset/' + dsets[idx] + '_' + str(dnum[idx]) + '.txt', 'w') as f:    
        for i in range(num_frames):
            num_ped = len(ds.video_position_matrix[i])
            for j in range(num_ped):
                line = str(i)
                line += ' ' + str(ds.video_pedidx_matrix[i][j])
                line += ' ' + str(ds.video_position_matrix[i][j][0])
                line += ' ' + str(ds.video_position_matrix[i][j][1])
                line += '\n'
                f.write(line)
