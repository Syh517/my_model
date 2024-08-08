from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.convolution_based import RocketClassifier
from sktime.classification.deep_learning.tapnet import TapNetClassifier
import numpy as np
from scipy import io
import time
import pandas as pd
import torch
from Classifiers.hydra import Hydra, SparseScaler

from Classifiers.metric import metric



Result_log_folder = './Example_Results_of_OS_CNN_for_multivariate/'
dataset_path='./ServerMachineDataset/'
save_path = './ServerMachineDataset/save2/'
dataset_name='ServerMachineDataset'
classifier_save_path = './explainer/kmeans.pkl'

dataset='MBA' #MBA MSL PSM SMAP SWaT SM
model_save_path = Result_log_folder + 'trained_model_'+dataset


if __name__ == '__main__':
    print(dataset)
    variables = io.loadmat('./Data/'+dataset+'Dataset/train_'+dataset+ '.mat')
    X1_train = variables['A'] #读取时序数据X_train
    y_train = variables['C'][0] #读取标签y_train

    X1_train = np.float32(X1_train)
    y_train = np.int64(y_train)

    variables = io.loadmat('./Data/'+dataset+'Dataset/test_'+dataset+ '.mat')
    X1_test = variables['A'] #读取时序数据X_test
    y_test = variables['C'][0] #读取标签y_test

    X1_test=np.float32(X1_test)
    y_test = np.int64(y_test)



    print('train data 1 shape', X1_train.shape)
    print('train label shape', y_train.shape)
    print('test data 1 shape', X1_test.shape)
    print('test label shape', y_test.shape)
    print('unique train label', np.unique(y_train))
    print('unique test label', np.unique(y_test))

    begin_time = time.time()


    # transform = Hydra(X1_train.shape[-1])
    #
    # X_training_transform = transform(torch.from_numpy(X1_train).float().unsqueeze_(1))
    # X_test_transform = transform(torch.from_numpy(X1_test).float().unsqueeze_(1))
    # scaler = SparseScaler()
    # X_training_transform = scaler.fit_transform(X_training_transform)
    # X_test_transform = scaler.transform(X_test_transform)
    # model = RocketClassifier(rocket_transform="rocket")
    # model.fit(X_training_transform.detach().cpu().numpy(), y_train)


    # model = MultiRocketHydraClassifier(n_kernels=100)
    model = RocketClassifier(num_kernels=10000, rocket_transform="rocket")
    model.fit(X1_train,y_train)

    end_time = time.time()
    exe_time = round((end_time - begin_time) / 60, 2)
    print(exe_time)

    y_predict = model.predict(X1_test)
    # y_predict = model.predict(X_test_transform.detach().cpu().numpy())
    y_predict = np.int64(y_predict)

    t = metric(y_test, y_predict)
    print('Precision:', t['Precision'], '\nRecall:', t['Recall'], '\nF1:', t['F1'], '\nAccuracy:', t['Accuracy'], '\nAUC:', t['AUC'])

