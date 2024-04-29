from Classifiers.OS_CNN_2.OS_CNN_easy_use_2 import OS_CNN_easy_use as TwoNet_MS_CAM
from Classifiers.OS_CNN_2.OS_CNN_res_easy_use_2 import OS_CNN_easy_use as TwoNet_res_MS_CAM
from Classifiers.OS_CNN_3.OS_CNN_easy_use_3 import OS_CNN_easy_use as OneNet_Concat
from Classifiers.OS_CNN_3.OS_CNN_res_easy_use_3 import OS_CNN_easy_use as OneNet_res_Concat
from Classifiers.OS_CNN_4.OS_CNN_easy_use_4 import OS_CNN_easy_use as OneNet_MS_CAM
from Classifiers.OS_CNN_4.OS_CNN_res_easy_use_4 import OS_CNN_easy_use as OneNet_res_MS_CAM

import numpy as np
from scipy import io
from Classifiers.metric import metrics

Result_log_folder = './Example_Results_of_OS_CNN_for_multivariate/'
dataset_path='./ServerMachineDataset/'
save_path = './ServerMachineDataset/save/'
dataset_name='ServerMachineDataset'


if __name__ == '__main__':
    variables = io.loadmat(save_path + 'machine-1-8.mat')
    X1_train = variables['A'] #读取时序数据X_train
    X2_train = variables['B'] #读取MTF矩阵X_trian
    y_train = variables['C'][0] #读取标签y_train

    X1_train = np.float32(X1_train)
    X2_train = np.float32(X2_train)
    y_train = np.int64(y_train)

    variables = io.loadmat(save_path + 'machine-2-4.mat')
    X1_test = variables['A'] #读取时序数据X_test
    X2_test = variables['B']  # 读取MTF矩阵X_test
    y_test = variables['C'][0] #读取标签y_test

    X1_test=np.float32(X1_test)
    X2_test = np.float32(X2_test)
    y_test = np.int64(y_test)



    print('train data 1 shape', X1_train.shape)
    print('train data 2 shape', X2_train.shape)
    print('train label shape', y_train.shape)
    print('test data 1 shape', X1_test.shape)
    print('test data 2 shape', X2_test.shape)
    print('test label shape', y_test.shape)
    print('unique train label', np.unique(y_train))
    print('unique test label', np.unique(y_test))

    # creat model and log save place

    # 2
    # TwoNet_MS_CAM
    # TwoNet_res_MS_CAM
    # 3
    # OneNet_Concat
    # OneNet_res_Concat
    # 4
    # OneNet_MS_CAM
    # OneNet_res_MS_CAM

    model = OneNet_res_Concat(
        Result_log_folder=Result_log_folder,  # the Result_log_folder
        dataset_name=dataset_name,  # dataset_name for creat log under Result_log_folder
        device="cuda:0",  # Gpu
        max_epoch=500,
        # In our expirement the number is 2000 for keep it same with FCN for the example dataset 500 will be enough
        paramenter_number_of_layer_list=[8 * 128 * X1_train.shape[1], 5 * 128 * 256 + 2 * 256 * 128] #两种X_train参数是一样的
    )

    model.fit(X1_train,X2_train, y_train, X1_test, X2_test,y_test)

    y_predict = model.predict(X1_test,X2_test)

    # print('correct:', y_test)
    # print('predict:', y_predict)
    t = metrics(y_test, y_predict)
    print('Precision:',t['Precision'],'\nRecall:',t['Recall'],'\nF1:',t['F1'],'\nAccuracy:',t['Accuracy'])
