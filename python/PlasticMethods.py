#-*- coding: utf-8 -*-

import os.path
import sys
import codecs
import numpy
import errno

#######################
######## Paths ########
#######################

def getProjectFolder():
    ProjectFolder = os.path.join(os.path.expanduser('~'),'PlasticRings')   
    return ProjectFolder

def MakeLogFile(filename):
    halfpath=os.path.join(getProjectFolder(), 'logs')
    fullpath = os.path.join(halfpath, filename)
    return fullpath

def MakeModelPath(filename):
    halfpath=os.path.join(getProjectFolder(), 'models')
    fullpath = os.path.join(halfpath, filename)
    return fullpath

def RawPicturesPath(defective=True):
    halfpath = os.path.join(getProjectFolder(),'Samples', "Organized_Subfolders")
    if defective:
        fullpath = os.path.join(halfpath,'Defective')
    else:
        fullpath = os.path.join(halfpath,'Nondefective')
    return fullpath

def getDataPath(mode = 'grayscale_crop_600_resize_240'):
    data_path = os.path.join(getProjectFolder(),'Samples', 'numpy', '{}_vector.npy'.format(mode))
    if not os.path.exists(data_path):
        raise ValueError('Invalid mode = {}'.format(mode))
    return data_path

def getProcessedPath(mode = 'grayscale_crop_600_resize_240'):
    folder_path = os.path.join(getProjectFolder(),'Samples', 'Processed', '{}'.format(mode))
    return folder_path

### To check the tensorboard log
def print_log_instructions():
    print("To be able to see Tensorboard in your local machine after training on a server")
    print("    1. exit current server session")
    print("    2. connect again with the following command:")
    print("        ssh -L 16006:127.0.0.1:6006 -p 2222 ealeman@133.44.109.144")
    print("    3. execute in terminal")
    print("        tensorboard --logdir='{}'".format(MakeLogFile('')))
    print("    4. on local machine, open browser on:")
    print("        http://127.0.0.1:16006")

##################################
######## NN Training Data ########
##################################

def OneHot(Y):
    uniqueY = numpy.unique(Y)
    oneHotY = numpy.zeros([Y.shape[0], uniqueY.shape[0]])
    for num, i in enumerate(Y):
        oneHotY[num][i] = 1
    return oneHotY

# get X, Y, test_x, test_y
def ReadyData(data, test_size = 1000, do_shuffle=True):
    if do_shuffle:
        numpy.random.shuffle(data)
    train_data = data[:-test_size]
    test_data = data[-test_size:]
    X,Y = zip(*train_data)
    test_x, test_y = zip(*test_data)
    X = numpy.array(list(X))#,dtype=object
    Y = numpy.array(list(Y))
    test_x = numpy.array(list(test_x))#,dtype=object
    test_y = numpy.array(list(test_y))
    return X,Y,test_x,test_y

def LoadData(mode='grayscale_crop_600_resize_240',split_test=True, test_size=1000, do_shuffle=True):
    print("Loading numpy array from file")
    # ## Load after first time
    input_path = getDataPath(mode = mode)
    if os.path.exists(input_path):
        data = numpy.load(input_path)
        print("Done")
        if split_test:
            X,Y,test_x,test_y = ReadyData(data, test_size=test_size, do_shuffle=do_shuffle)
            return X,Y,test_x,test_y
        else:
            return data
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_path)

# # Test NN training output
# def getCurrentAverageError(model,test_x,test_y):
#     pred_y = [p[0] for p in model.predict(test_x)]
#     losses = [(i[0]-i[1])**2 for i in zip(pred_y,test_y)]
#     mean_square_man = numpy.average(losses)
#     av_error = mean_square_man**0.5
#     return av_error

if __name__ == '__main__':
    pass
