import numpy as np
import  random


def constuctinput(numberOfUser, numberOfItem, mtx_np,neighbor_user, k_user,
                          neighbor_item, k_item):
    X_trains, X_tests, y_trains, y_tests =[],[],[],[]
    y_new = []
    x_new = []
    count = 0
    for id in range(1, numberOfUser):
        # Import rating information of all neighbors of users into X_np
        X = []
        tmpmean = []
        for neighbor_id in neighbor_user[id]:
            X.append(mtx_np[neighbor_id])
        X_np = np.array(X, dtype=float)
        X_np = np.reshape(X_np, (k_user, numberOfItem))

        # Store user rating information
        y = mtx_np[id]
        y = np.reshape(y, (1, numberOfItem))

        # Transpose, each line denotes the rating information of all neighbors
        X_np = X_np.T
        y = y.T

        # Pick out items that have been rated by user u
        origine_index = []  # Record the original index
        for keys in range(numberOfItem):
            if y[keys] != 0:
                temp = []
                tmp = []
                for k in range((X_np[keys].shape[0])):
                    tmprating = X_np[keys][k]
                    if(tmprating==0):
                        for node_neighbor_item in neighbor_item[keys]:
                            if (X_np[node_neighbor_item][k] != 0):
                                tmprating = X_np[node_neighbor_item][k]
                                break

                    tmp.append(tmprating)
                temp.extend(tmp)
                # Add the rating information of neighbor items
                for neighbor_id in neighbor_item[keys]:
                    tmprating = mtx_np[id][neighbor_id]
                    if(mtx_np[id][neighbor_id] == 0):
                        for node_neighbor_id in neighbor_item[neighbor_id]:
                            if(mtx_np[id][node_neighbor_id] != 0):
                                tmprating = mtx_np[id][node_neighbor_id]
                                break
                    temp.append(tmprating)
                x_new.append(temp)
                y_new.extend(y[keys])
                origine_index.append(keys)
                count += 1
        # Convert list to array form for easy training
        # Split data into training and test sets

    y_new = np.reshape(y_new, (count, 1))
    x_new = np.reshape(x_new, (count, k_user + k_item))
    X_train, X_test, y_train, y_test, indexs = data_spilit(x_new, y_new, 0.8)

    X = X_train.tolist()
    Y = X_test.tolist()
    X_tmp = []
    Y_tmp = []
    for i in range(len(X)):
        user_tmp = []
        item_tmp = []
        for j in range(k_user):
            user_tmp.append(X[i][j])
            item_tmp.append(X[i][j + k_user])
        user_tmp.extend(item_tmp)
        X_tmp.append(user_tmp)

    for i in range(len(Y)):
        user_tmp = []
        item_tmp = []
        for j in range(k_user):
            user_tmp.append(Y[i][j])
            item_tmp.append(Y[i][j + k_user])
        user_tmp.extend(item_tmp)
        Y_tmp.append(user_tmp)

    X_trains.extend(X_tmp)
    X_tests.extend(Y_tmp)
    y_trains.extend(y_train.tolist())
    y_tests.extend(y_test.tolist())


    X_trainsM = np.reshape(X_trains, (len(X_trains),len(X_trains[0])))
    X_testsM = np.reshape(X_tests, (len(X_tests), len(X_tests[0])))
    y_trainsM = np.reshape(y_trains, (len(y_trains), len(y_trains[0])))
    y_testsM = np.reshape(y_tests, (len(y_tests), len(y_tests[0])))
    return X_trainsM, X_testsM, y_trainsM, y_testsM

def data_spilit(x, y, pct):
    # x：input data
    # y：the label
    # pct：the percent of training data
    length = len(y)
    index = int(length * pct)
    indexes = np.array(range(0, length))
    random.shuffle(indexes)
    trn_idxes = indexes[0:index]
    tst_idxes = indexes[index:length]
    # print(trn_idxes)
    # print(x[trn_idxes,:])
    x_train = x[trn_idxes, :]
    x_test = x[tst_idxes, :]
    y_train = y[trn_idxes, :]
    y_test = y[tst_idxes, :]
    return x_train, x_test, y_train, y_test, indexes

