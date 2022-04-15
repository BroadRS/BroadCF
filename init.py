import GetKNearestNeighbor
from evaluations import RMSE_label
from readerdata import *
from Datapreprocessing import *
from broadlearningsystem import *
from sklearn import metrics

if __name__ == "__main__":
    N1 = 10  # # of nodes belong to each window
    N2 = 10  # # of windows -------Feature mapping layer
    N3 = 500  # # of enhancement nodes -----Enhance layer
    L = 5  # # of incremental steps
    M1 = 50  # # of adding enhance nodes
    s = 0.8  # shrink coefficient
    C = 2 ** -30  # Regularization coefficient
    ar = 1
    dataFile = './transratings.csv'
    map_num= 25
    enhance_num = 25
    map_batchsize = 15
    enh_batchsize= 10
    k_user = 5
    k_item = 5
    EPOCH = 2

    numberOfUser, numberOfItem, mtx_np =readamazon2(dataFile)
    neighbor_user = GetKNearestNeighbor.k_neighbors(mtx_np, k_user, numberOfUser)
    neighbor_item = GetKNearestNeighbor.k_neighbors(mtx_np.T, k_item, numberOfItem)
    print("data reader complete")
    traindata, testdata, trainlabel, testlabel = predict_user_and_item(numberOfUser,numberOfItem,mtx_np,neighbor_user, k_user,neighbor_item, k_item)
    trainlabel = trainlabel.flatten()
    testlabel = testlabel.flatten()
    bls = broadNet(map_num=map_num,  # 初始时多少组mapping nodes
                   enhance_num=enhance_num,  # 初始时多少enhancement nodes
                   EPOCH=EPOCH,  # 训练多少轮
                   map_function='relu',
                   enhance_function='relu',
                   map_batchsize=map_batchsize,  # 每一组的神经元个数
                   enh_batchsize=enh_batchsize,
                   DESIRED_ACC=0.95,  # 期望达到的准确率
                   STEP=int(1)  # 一次增加多少组enhancement nodes
                   )
    labelunique = {}

    num = 0
    for i in range(len(trainlabel)):
        if trainlabel[i] not in labelunique:
            labelunique.update({trainlabel[i]: num})
            num += 1
    labels = sorted(labelunique)


    starttime = datetime.datetime.now()
    bls.fit(traindata, trainlabel)
    endtime = datetime.datetime.now()
    runtime = str((endtime - starttime).total_seconds())
    print('the training time of BLS is {0} seconds'.format((endtime - starttime).total_seconds()))

    pre = bls.predict(testdata)
    teststarttime = datetime.datetime.now()
    predictlabel = bls.weightPredict(testdata)
    testendtime = datetime.datetime.now()
    testtime = str((testendtime - teststarttime).total_seconds())

    # predictlabel - ranges[0]
    # print(metrics.mean_absolute_error(testlabel, predictlabel))
    lista = []
    for i in range(len(pre)):
        lista.append(labels[pre[i]])
    mae = str(metrics.mean_absolute_error(testlabel, lista))
    rmse = str(RMSE_label(pre, testlabel, labels))
    print(metrics.mean_absolute_error(testlabel, lista))
    print(RMSE_label(pre, testlabel, labels))


