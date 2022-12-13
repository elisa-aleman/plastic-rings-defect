import os.path
import sys
PythonPath = os.path.join(os.path.expanduser('~'),'PlasticRings', 'python')
sys.path.append(os.path.abspath(PythonPath))
from PlasticMethods import *
import numpy
import cv2

def main():
    modelist = [
                'grayscale_crop_600_resize_240',
                'edges_crop_600_resize_240'
                ]
    for mode in modelist:
        print('Vectorization of files from mode = {}'.format(mode))
        folder_path = getProcessedPath(mode = mode)
        frame_paths = os.listdir(folder_path)
        frame_paths = sorted(frame_paths)
        Y = numpy.array([int(p.split('defective_', 1)[1].split('.jpg')[0]) for p in frame_paths])
        Y = OneHot(Y)
        frame_fullpaths = [os.path.join(folder_path, fp) for fp in frame_paths]
        print('Creating empty numpy')
        data = numpy.empty((len(frame_fullpaths),2), dtype=object)
        print('Appending to numpy')
        maxlen = len(frame_fullpaths)
        for num, ffp in enumerate(frame_fullpaths):
            print('{} : {} of {} appended to list'.format(mode,num, maxlen))
            data[num][0] = cv2.imread(ffp, cv2.IMREAD_UNCHANGED)
            data[num][1] = Y[num]
        npy_path = os.path.join(os.path.expanduser('~'), 'PlasticRings','Samples', 'numpy', '{}_vector.npy'.format(mode))
        print('Saving File...')
        numpy.save(npy_path,data)
        print('Done')

if __name__ == '__main__':
    main()

