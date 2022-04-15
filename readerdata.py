import csv
from datetime import datetime
import scipy.sparse as sparse
import numpy as np
import xlrd
import pandas as pd
def unique(old_list):
    # Minimize the use_id and item_id in the dataset
    count = 1
    dic = {}
    for i in range(len(old_list)):
        if old_list[i] in dic:
            old_list[i] = dic[old_list[i]]
        else:
            dic[old_list[i]] = count
            old_list[i] = count
            count += 1
    return old_list



def readData(dataset = './ratings.csv'):
    with open(dataset) as f:
        reader=csv.reader(f)
        header_row=next(reader)
        row, column, value, timestamp= [], [], [],[]
        for data in reader:
            current_date = int(data[0])
            row.append(current_date)


            col = int(data[1])
            column.append(col)

            valu = int(float(data[2]))
            value.append(valu)

            time = (data[3])
            timestamp.append(time)
            # if(count>=5000):
            #     break
            # count+=1

    row = unique(row)
    column = unique(column)
    numberOfUser = max(row)+1
    numberOfItem = max(column)+1

    column = unique(column)
    mtx = sparse.coo_matrix((value, (row, column)), shape=(numberOfUser, numberOfItem))
    mtx = mtx.todense()
    mtx = np.array(mtx)
    # mdx = np.zeros([mtx.shape[0], mtx.shape[1]])
    # mean_user = []
    # for i in range(mtx.shape[0]):
    #     temp = mtx[i]
    #     if len(temp[temp != 0]) == 0:
    #         mean_user.append(0)
    #     else:
    #         mean_user.append(int(np.mean(temp[temp != 0])))
    #
    # for i in range(mtx.shape[0]):
    #    for j in range(mtx.shape[1]):
    #        if mtx[i][j] != 0:
    #            mdx[i][j] = mtx[i][j]
    #        else:
    #            mdx[i][j] = mean_user[i]
    return numberOfUser, numberOfItem, mtx

def readamazon2(dataset = './mllm-ratings.csv'):
    with open(dataset) as f:
        reader=csv.reader(f)
        header_row=next(reader)
        row, column, value, timestamp= [], [], [],[]
        for data in reader:
            current_date = int(data[1])
            row.append(current_date)
            col = int(data[2])
            column.append(col)

            valu = int(float(data[3]))
            value.append(valu)

            time = (data[4])
            timestamp.append(time)
            # if(count>=25000):
            #     break

    row = unique(row)
    column = unique(column)
    numberOfUser = max(row)+1
    numberOfItem = max(column)+1

    column = unique(column)
    mtx = sparse.coo_matrix((value, (row, column)), shape=(numberOfUser, numberOfItem))
    mtx = mtx.todense()
    mtx = np.array(mtx)
    return numberOfUser, numberOfItem, mtx


def truedata(dataset = './ratings_Amazon_Instant_Video.csv'):
    user = {}
    item = {}
    numberOfUser = 0
    numberOfItem = 0
    row, column, value, timestamp = [], [], [], []
    count = 0
    with open(dataset) as f:
        reader=csv.reader(f)
        header_row = next(reader)
        for data in reader:
            current_date = (data[0])
            if(current_date) not in user:
                user.update({current_date:numberOfUser})
                numberOfUser +=1

            row.append(user[current_date])


            col = (data[1])
            if (col) not in item:
                item.update({col: numberOfItem})
                numberOfItem += 1
            column.append(item[col])

            #
            valu = int(float(data[2]))
            value.append(valu)

            print(user[current_date], item[col], valu)
            #
            if(count >= 15000):
                break
            count+=1
            # time = (data[3])
            # timestamp.append(time)
    row = unique(row)
    column = unique(column)
    numberOfUser +=1
    numberOfItem +=1
    column = unique(column)
    mtx = sparse.coo_matrix((value, (row, column)), shape=(numberOfUser, numberOfItem))
    mtx = mtx.todense()
    mtx = np.array(mtx)
    # mdx = np.zeros([mtx.shape[0], mtx.shape[1]])
    # mean_user = []
    #
    # for i in range(mtx.shape[0]):
    #     temp = mtx[i]
    #     if len(temp[temp != 0]) == 0:
    #         mean_user.append(0)
    #     else:
    #         mean_user.append(np.mean(temp[temp != 0]))
    #
    # for i in range(mtx.shape[0]):
    #     for j in range(mtx.shape[1]):
    #         if mtx[i][j] != 0:
    #             mdx[i][j] = mtx[i][j]
    #         else:
    #             mdx[i][j] = mean_user[i]
    # # mdx = mtx
    return  numberOfUser, numberOfItem, mtx


def readerxls(dataset = './jester-data-2.xls') :
    data = pd.read_excel(dataset)
    var1 =np.array(data)
    biglist = []
    for i in range(len(var1)):
        smalllist=[]
        for j in range(len(var1[0])):
            if(j==0):
                continue
            elif(var1[i][j]==99):
                smalllist.append(0)
            else:
                if(int(var1[i][j]) == 0):
                    if(var1[i][j]>0):
                        var1[i][j] = 1
                    if (var1[i][j] < 0):
                        var1[i][j] = -1
                    if(var1[i][j] == 10):
                        var1[i][j] = 9
                smalllist.append((int(var1[i][j])))
        biglist.append( smalllist)
        # if(i>90000):
        #     break
    mtx = np.array(biglist)
    mtx = np.array(mtx)
    numberOfUser = mtx.shape[0]
    numberOfItem = mtx.shape[1]
    # print(mtx)
    # print(mdx)
    # print(numberOfUser,numberOfItem)
    return numberOfUser, numberOfItem, mtx

if __name__ == "__main__":
    readamazon2('./transratings.csv')

