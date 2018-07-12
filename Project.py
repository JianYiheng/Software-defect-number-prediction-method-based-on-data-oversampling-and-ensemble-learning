import os
import math

import numpy as np
import pandas as pd

from time import time
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import NearestNeighbors

# path = os.getcwd()
# name = os.listdir(path+'\\Data')
# for i in name:
#     subpath = path +'\\Data' + '\\' + i
#     subname = os.listdir(subpath)
#     for j in subname:
#         finalpath = subpath + '\\' + j
#         x = pd.read_csv(finalpath,engine='python').dropna(how='all')
#         newpath = path+'\\data'+'\\'+i+'\\'+j
#         print(newpath)
#         x.to_csv(newpath)

'''
***************************************************************
    功能：获取原始模块数据：
    输出：一个二维列表，分别模块列表名称和相应的DataFrame格式表格
    备注：默认数据都在当前文件夹的data文件夹里面，data文件夹下面有ant,
    jedit,ivy,synapse和xalan五个文件夹，分别储存五种数据的csv文件。
    
    
    Menu:
        Project.py
        Data
            -ant
                -ant-1.3.csv
                -ant-1.4.csv
                -...
            -jedit
            -ivy
            -....
***************************************************************
'''
def get_data():
    path = os.getcwd()+'\\data'
    name = os.listdir(path)
    name0 = []
    output = []
    for i in name:
        if i in ['ant','jedit','ivy','synapse','xalan']:
            sub_path = path+'\\'+i
            for j in os.listdir(sub_path):
                data = pd.read_csv(sub_path+'\\'+j,engine='python')
                data = data.iloc[:,5:]
                output.append(data)
                name0.append(j)
    return name0,output

'''
**************************************************************
    功能：模块数据提取函数        
    输入：modules-模块，Type-输出类型选择
    输出：
        当Type==-1时，输出缺陷模块，缺陷模块的特征和缺陷模块的Bug；
        当Type==0时，输出出模块的特征和模块的Bug；
        当Type==1时，输出正常模块，正常模块的特征和正常模块的Bug；
**************************************************************
'''
def seperateData(modules, Type):
    if Type==-1:
        rare_modules = modules[modules.bug!=0]
        rare_char = rare_modules.iloc[:, :-1]
        rare_bug = rare_modules.iloc[:, -1]
        return rare_modules, rare_char, rare_bug
    elif Type==1:
        normal_modules = modules[modules.bug==0]
        
        normal_char = normal_modules.iloc[:, :-1]
        return normal_modules, normal_char
    else:
        char = modules.iloc[:, :-1]
        bug = modules.iloc[:, -1]
        return char, bug

'''
***************************************************************
    功能：回归预测函数
    输入：trainChar-训练集模块特征，trainBug-训练集模块Bug
          testChar-测试集模块特征，choose-回归类型选择（缺省为0）
    输出：
        当choose=0时，输出决策树回归预测bug数量序列
        当choose=1时，输出线性回归预测结果bug数量序列
        当choose=-1时，输出贝叶斯回归预测结果bug数量序列
***************************************************************
'''
def Regression(trainChar,trainBug, testChar, choose=0):
    if choose == 0:
        dtr = DecisionTreeRegressor().fit(trainChar,trainBug)
        return dtr.predict(testChar).astype(int)
    elif choose == 1:
        lr = linear_model.LinearRegression().fit(trainChar,trainBug)
        return lr.predict(testChar).astype(int)
    else:
        bayes = BayesianRidge().fit(trainChar,trainBug)
        return bayes.predict(testChar).astype(int)


'''
***************************************************************
    功能：FPA计算函数
    输入：testBug-模块Bug实际值，testPre-模块Bug预测值
    输出：FPA值
***************************************************************
'''
def FPA(testBug, testPre):
    K = len(testBug)
    N = np.sum(testBug)
    
    sort_axis = np.argsort(testPre)
    testBug=np.array(testBug)
    testBug = testBug[sort_axis]
    P = sum(np.sum(testBug[m:])/N for m in range(K+1))/K
    return P

'''
***************************************************************
    功能：SMOTE过采样函数
    输入：modules_input-模块，ratio-期望比值，即最终正常与缺陷模块数量的期望比值（缺省为1）
    输出：modules_new-采样后的平衡模块数据集
***************************************************************
'''
def smote(modules_input, ratio=1):
    modules, char, bug = seperateData(modules_input, -1)
    normal_modules, normal_char = seperateData(modules_input, 1)
    
    n = round((ratio*normal_modules.shape[0]-modules.shape[0])/modules.shape[0])
    k = 5
    
    if n<=0:
        return modules_input
    
    # 训练模型，取邻近的k个点（可修改邻近点数）
    neigh = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', n_jobs=-1)
    neigh.fit(char)
    index = neigh.kneighbors(n_neighbors=k, return_distance=False)
    # result结果为narray类型的索引矩阵
    a, b = index.shape

    # 此处的用法详见书P83
    axis0, axis1 = np.ogrid[:a, :b]
    sort_axis = np.zeros(b,dtype=int)
    for i in range(a):
        temp = np.arange(b)
        # 从k个邻近中随机抽取n个邻近
        np.random.shuffle(temp)
        sort_axis = np.vstack((sort_axis,temp))
    # index_rand就是最终过采样得到矩阵的 下标
    sort_axis = sort_axis[1:]
    index_rand = index[axis0, sort_axis]

    flag = 0
    new_list = []

    for i in range(a):
        for j in range(n):
            
            p = index_rand[i][np.random.randint(0,k)]

            # p = index_rand[i][j]
            # 计算新的模块的各项特征
            new = char.iloc[i]+(char.iloc[p]-char.iloc[i])*np.random.rand()
            #计算原两个模块与新模块之间的欧氏距离
            d1 = np.linalg.norm(new-char.iloc[i])
            d2 = np.linalg.norm(new-char.iloc[p])
            if d1 == 0 and d2 == 0:
                break
            # 计算新模块的缺陷个数
            bug_new = (d2*modules.iloc[i].loc['bug']+d1*modules.iloc[p].loc['bug'])/(d1+d2)
            bug_new = float(round(bug_new))
            # 将新模块的各项特征和缺陷个数合并
            new['bug'] = bug_new
            new_list.append(new)
            flag += 1
    # 将缺陷模块数据集和正常模块数据集合并

    modules = pd.concat([modules,pd.concat(new_list,axis=1).T],axis=0)
    # modules_new的样式分为三部分，最上面时旧的缺陷数据集，中间时新合成的缺陷数据集，下面时正常数据集
    modules_new = pd.concat([modules, normal_modules], axis=0)
    # modules_new = modules_new.dropna(axis=0) 
    return modules_new

'''
***************************************************************
    功能：数据过采样、回归预测并集成学习函数
    输入；dataset-软件模块，n-过采样次数，m-回归类型(参见Regression注释)
    输出：预测结果的FPA值
***************************************************************
'''
def Deposite_smote(dataset,n,m):
    rare_modules, rare_char, rare_bug = seperateData(dataset, -1)
    normal_modules, normal_char = seperateData(dataset, 1)
    rare_test_len = math.ceil(rare_modules.shape[0]/10)
    normal_test_len = math.ceil(normal_char.shape[0]/10)
    temp = np.array([])
    
    dataset_temp = []
    # 十折交叉验证，循环十次
    for i in range(10):
        rareX0 = i*rare_test_len
        rareX1 = (i+1)*rare_test_len
        normalX0 = i*normal_test_len
        normalX1= (i+1)*normal_test_len
        
        if rareX1 >=rare_modules.shape[0]:
            rareX1 = rare_modules.shape[0]
        if normalX1 >= normal_modules.shape[0]:
            normalX1 = normal_modules.shape[0]
        # 测试集， 取1/10的数据集
        testMod = pd.concat([rare_modules.iloc[rareX0:rareX1], normal_modules.iloc[normalX0:normalX1]],axis=0)
        # 训练集，取剩下的数据集
        trainMod = pd.concat([rare_modules.drop(rare_modules.index[list(range(rareX0,rareX1))]),normal_modules.drop(normal_modules.index[list(range(normalX0,normalX1))])],axis=0)
        dataset_temp.append(testMod)
        # 用以储存测试集预测Bug数据集
        testPre_sum = np.zeros(len(testMod))

        flag = 0
        # 每一折进行20次过采样
        for j in range(n):
            # 对训练集进行过采样
            trainMod_new = smote(trainMod,1)
            trainMod_new = trainMod_new.dropna(axis=0)
            # 提取 特征值 和 bug值
            trainChar, trainBug = seperateData(trainMod_new, 0)
            testChar, testBug = seperateData(testMod, 0)

            testPre = Regression(trainChar, trainBug, testChar,m)
            flag += 1
            testPre_sum += testPre
        # 对预测结果求平均
        testPre_sum /= flag
        testPre_sum = np.round(testPre_sum)
        temp = np.hstack((temp, testPre_sum))
    dataset_new = pd.concat(dataset_temp,axis=0)
    dataset_new['bug_new'] = temp
    return dataset_new

def Deposite_normal(dataset,n,m):
    rare_modules, rare_char, rare_bug = seperateData(dataset, -1)
    normal_modules, normal_char = seperateData(dataset, 1)
    rare_test_len = math.ceil(rare_modules.shape[0]/10)
    normal_test_len = math.ceil(normal_char.shape[0]/10)
    temp = np.array([])
    
    dataset_temp = []
    # 十折交叉验证，循环十次
    for i in range(10):
        rareX0 = i*rare_test_len
        rareX1 = (i+1)*rare_test_len
        normalX0 = i*normal_test_len
        normalX1= (i+1)*normal_test_len
        
        if rareX1 >=rare_modules.shape[0]:
            rareX1 = rare_modules.shape[0]
        if normalX1 >= normal_modules.shape[0]:
            normalX1 = normal_modules.shape[0]
        # 测试集， 取1/10的数据集
        testMod = pd.concat([rare_modules.iloc[rareX0:rareX1], normal_modules.iloc[normalX0:normalX1]],axis=0)
        # 训练集，取剩下的数据集
        trainMod = pd.concat([rare_modules.drop(rare_modules.index[list(range(rareX0,rareX1))]),normal_modules.drop(normal_modules.index[list(range(normalX0,normalX1))])],axis=0)
        dataset_temp.append(testMod)
        # 用以储存测试集预测Bug数据集
        testPre_sum = np.zeros(len(testMod))

        flag = 0
        # 每一折进行20次过采样
        for j in range(n):
            # 提取 特征值 和 bug值
            trainChar, trainBug = seperateData(trainMod, 0)
            testChar, testBug = seperateData(testMod, 0)

            testPre = Regression(trainChar, trainBug, testChar,m)
            flag += 1
            testPre_sum += testPre
        # 对预测结果求平均
        testPre_sum /= flag
        testPre_sum = np.round(testPre_sum)
        temp = np.hstack((temp, testPre_sum))
    dataset_new = pd.concat(dataset_temp,axis=0)
    dataset_new['bug_new'] = temp
    return dataset_new

x = get_data()[1][1]
z0 = Deposite_smote(x,5,0)
z1 = Deposite_normal(x,5,0)
result = []
for z in [z0, z1]:
    z_values = z.bug.values
    z_values = list({}.fromkeys(z_values).keys())
    z_values.sort()
    print(z_values)
    subresult = []
    for i in z_values:
        temp = z[z.bug == i]
        # print(temp.bug, temp.bug_new)
        diff = np.abs(temp.bug - temp.bug_new)
        output = np.power(diff, 2).sum()
        print(output)
        subresult.append(output)
    subresult = np.array(subresult)
    result.append(subresult)
rate = (result[1] - result[0])/result[1]
print(rate)