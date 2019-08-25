import numpy as np
import gzip
import pickle
import random
from PIL import Image

def load_data():
    '''加载数据集'''
    # 需为.gz压缩，gzip.open()打不开.zip
    f = gzip.open('F:/jupyter/mnist.pkl.gz', 'rb')
    # pickle模块可以序列化对象并保存到磁盘中，并在需要的时候读取出来，
    # 任何对象都可以执行序列化操作
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return(training_data, validation_data, test_data)

def normalize(data):
    """图片像素二值化，变成0-1分布"""
    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] != 0:
                data[i][j] = 1
            else:
                data[i][j] = 0
    return data

def pca(data, top_n_feature):
    '''用PCA对数据矩阵降维'''
    # print(data.shape[0],data.shape[1])
    # 数据中心化，即训练数据代表的矩阵减去它的均值
    mean_vals = data.mean(axis = 0)
    changed_data = data - mean_vals
    # print(mean_vals.shape,changed_data.shape)
    # 计算协方差矩阵
    cov_mat = np.cov(changed_data, rowvar=0)
    # print(cov_mat.shape)
    # 计算特征值和特征向量
    # 注意，此处取出的特征向量为复数形式，且经过观察发现，特征向量的虚部都为0j，
    feature_vals, all_feature_vects = np.linalg.eig(np.mat(cov_mat))
    # print(feature_vals.shape,all_feature_vects.shape)
    # 返回特征值从小到大排序的下标
    small_to_large_index = np.argsort(feature_vals)
    # print(small_to_large_index)
    # 将上述下标逆序
    large_to_small_index = small_to_large_index[::-1]
    # 取出前top_n_feature个特征值对应下标
    index_expected = large_to_small_index[:top_n_feature]
    # print(index_expected)
    # 取出对应特征向量
    feature_vects = all_feature_vects[:, index_expected]
    # print(feature_vects)
    # 取特征向量实部
    feature_vects = np.real(feature_vects)
    # 将数据转到新的空间
    low_dim_data = changed_data.dot(feature_vects)
    # 根据前几个特征向量重构回去的训练数据矩阵
    restructure_data = low_dim_data.dot(feature_vects.T) + mean_vals 
    # return low_dim_data, restructure_data
    return restructure_data

def blocks(data, blocks_per_img, m1, static):
    """该函数用来完成取块操作"""
    block = []
    for x in data:
        block = get_block(x, blocks_per_img, block, static)
    a = np.zeros((m1, 100))
    for i in range(m1):
        for k in range(100):
            a[i][k] = block[i][k]
    return a

def get_block(x, blocks_per_img, block, static):
    """该函数对单个图像完成取块并添加到列表操作"""
    x = x.reshape(28, 28)
    if static == 0:
        for k in range(blocks_per_img):
            i = np.random.randint(0, 17)
            j = np.random.randint(0, 17)
            y = x[i:i+10, j:j+10]
            y = y.ravel()
            y = np.array(y)
            block.append(y)
            print(np.array(block).shape)
    elif static == 1:
        i=0
        j=0
        for k in range(blocks_per_img):
            y = x[4+i:14+i, 4+j:14+j]
            y = y.ravel()
            block.append(y)
            i +=1
            j +=1
    return block
    
def cal_vects(data, points, block):
    """计算图片的词向量矩阵"""
    vects = np.zeros((len(data), 100))
    for i in range(len(data)):
        for j in range(10):
            index = 0
            min = np.sum((block[i*10+j] - points[0])**2)
            for k in range(1, len(points)):
                if np.sum((block[i*10+j] - points[k])**2) < min:
                    index = k
                    min = np.sum((block[i*10+j] - points[k])**2)
            vects[i][index] += 1
    return vects

def cal_accuracy(pred_test, test_labels):
    '''计算预测正确率'''
    num_correct = 0 
    for a, y in zip(pred_test, test_labels):
            if a == y:
                num_correct += 1
    return 1.0 * num_correct / len(test_labels)

def k_means(data, points, mat, labels, m, n, k):
    """K-means迭代函数"""
    # 计算每个样本到各中心的欧式距离
    for i in range(m):
        for j in range(k):
            count = 0
            for z in range(n):
                count += (data[i][z]-points[j][z])**2
            mat[i][j] = np.sqrt(count)
    # 为每个样本重新分类
    for i in range(m):
        index = np.argsort(mat[i])
        label = index[0]
        labels[i] = label
    # 计算每个中心的新坐标，用均值更新
    for i in range(k):
        a = [j  for j, x in enumerate(labels) if x == i]
        length = len(a)
        for y in range(n):
            count = 0
            for z in a:
                count += data[z][y]
            # 除0错误处理的很粗糙，所以才可能times上升而准确率下降
            points[i][y] =  count / (length+1)
    return points, mat