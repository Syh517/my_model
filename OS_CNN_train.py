import numpy as np
import time
from scipy import io

from Classifiers.OS_CNN.CNN_easy_use import OS_CNN_easy_use as CNN_res_easy_use
from Classifiers.OS_CNN.OS_CNN_res_easy_use import OS_CNN_easy_use as OS_CNN_res_easy_use
from Classifiers.metric import metric





Result_log_folder = './Example_Results_of_OS_CNN_for_multivariate/'
dataset_path='./ServerMachineDataset/'
save_path = './ServerMachineDataset/save2/'
dataset_name='ServerMachineDataset'


dataset='SMAP' #MBA MSL PSM SMAP SWaT SM
model_save_path = Result_log_folder + 'trained_model_'+dataset

if __name__ == '__main__':
    print(dataset)
    variables = io.loadmat('./Data/' + dataset + 'Dataset/train_' + dataset + '.mat')
    X_train = variables['A']  # 读取时序数据X_train
    # X_train = variables['B']  # 读取MTF矩阵X_trian
    y_train = variables['C'][0]  # 读取标签y_train

    variables = io.loadmat('./Data/' + dataset + 'Dataset/test_' + dataset + '.mat')
    X_test = variables['A']  # 读取时序数据X_test
    # X_test = variables['B']  # 读取MTF矩阵X_test
    y_test = variables['C'][0]  # 读取标签y_test


    X_train=np.float32(X_train)
    y_train=np.int64(y_train)
    X_test=np.float32(X_test)
    y_test = np.int64(y_test)



    print('train data shape', X_train.shape)
    print('train label shape', y_train.shape)
    print('test data shape', X_test.shape)
    print('test label shape', y_test.shape)
    print('unique train label', np.unique(y_train))
    print('unique test label', np.unique(y_test))

    begin_time = time.time()
    print(begin_time)

    # CNN_easy_use
    # OS_CNN_res_easy_use
    model = OS_CNN_res_easy_use(
        Result_log_folder=Result_log_folder,  # the Result_log_folder
        dataset_name=dataset_name,  # dataset_name for creat log under Result_log_folder
        device="cuda:0",  # Gpu
        max_epoch=500,
        # In our expirement the number is 2000 for keep it same with FCN for the example dataset 500 will be enough
        paramenter_number_of_layer_list=[8 * 128 * 1, 5 * 128 * 256 + 2 * 256 * 128] #两种X_train参数是一样的
    )

    model.fit(X_train, y_train, X_test, y_test)

    y_prob = model.predict(X_test)

    y_predict = []
    for yi in y_prob:
        yi = np.argmax(yi,axis=0)
        y_predict.append(yi)
    y_predict=np.array(y_predict)



    # print('correct:', y_test)
    # print('predict:', y_predict)
    # print('correct:', sum(y_test))
    # print('predict:', sum(y_predict))
    #
    t = metric(y_test, y_predict)
    print('Precision:', t['Precision'], '\nRecall:', t['Recall'], '\nF1:', t['F1'], '\nAccuracy:', t['Accuracy'],
          '\nAUC:', t['AUC'])

    end_time = time.time()
    print(end_time)
    exe_time = round((end_time - begin_time) / 60, 2)
    print(exe_time)



