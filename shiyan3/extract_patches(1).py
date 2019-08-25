from data_process import load_image
import numpy as np
import random
from PCA import pca
from K_means import K_Mean
import matplotlib.pyplot as plt
from Naive_Bayes_classifier import NaiveBayesClassifier
from sklearn.cluster import KMeans
from sklearn.svm import SVC
# 导入训练集，并把它分为训练集与验证集
images = load_image(0)
labels = load_image(1).astype(np.int)  # 标签转化为整型
images_num = images.shape[0]
# 根据9比1的比例分为验证集和测试集
train_num = int(images_num * 0.1)
train_images = images[:train_num]
train_labels = labels[:train_num]
val_images = images[train_num:]
val_labels = labels[train_num:]
# 导入测试集
test_images = load_image(2)
test_labels = load_image(3).astype(np.int)
test_num = test_images.shape[0]


# 数据集变形
# train_images = train_images.reshape(train_images.shape[0],-1)
# val_images = val_images.reshape(val_images.shape[0],-1)
# test_images = test_images.reshape(test_num.shape[0],-1)
# 上面为对数据进行处理的部分
# 接下来对数据进行随机取块,每一张图片取10块，共270000块

def Random_pitchs(X, pitch_num, pitch_size):  # 如果要改变y参数里面输入一个y
    '''
    :param X: 输入的图片矩阵 y：标签
    :param pitch_num: 每张图片随机取块的数量
    :param pitch_size: 每块的尺寸
    :return:
    '''
    result = np.zeros((X.shape[0] * pitch_num, pitch_size, pitch_size))
    for i in range(X.shape[0]):
        for j in range(pitch_num):
            # if j != 0:
            # y = np.r_[y,y[i]]
            num = random.randint(0, X[0].shape[0] - pitch_size - 1)
            result[i * pitch_num + j] = X[i][np.arange(num, num + pitch_size), np.arange(num, num + pitch_size)]
    return result  # ,y


def Pitch(X, pitch_num, pitch_size):
    '''
    固定步长为2 从28*28的块中切36块6*6，上下不重叠 从1,2位置开始切，其他起点与这个相对应
    :param X: 输入的图片矩阵
    :param pitch_num: 每张图片随机取块的数量
    :param pitch_size: 每块的尺寸
    :return:
    '''
    result = np.zeros((X.shape[0] * pitch_num, pitch_size, pitch_size))
    for i in range(X.shape[0]):
        num = 2
        for j in range(pitch_num // 9):
            # if j != 0:
            # y = np.r_[y, y[i]]  #取块的同时改变y
            for k in range(9):
                result[i * pitch_num + j * 9 + k] = X[i][np.arange(num + k * 2, num + k * 2 + pitch_size),  # 2为步长
                                                         np.arange(num + j * pitch_size,
                                                                   num + j * pitch_size + pitch_size)]
    return result  # ,y


# 取块
train_image_pitchs = Random_pitchs(train_images, 10, 6)
print(train_image_pitchs.shape, 'images num')
train_image_pitchs_second = Pitch(train_images, 36, 6)
print(train_image_pitchs_second.shape, '1')
test_image_pitchs = Pitch(test_images, 36, 6)


# 归一化，论文里写的美白化？whitening
def Normalization(X):
    X -= np.mean(X, axis=0)
    X /= np.std(X, axis=0)
    return X


train_image_pitchs_nor = Normalization(train_image_pitchs)  # 归一化
train_image_pitchs_second_nor = Normalization(train_image_pitchs_second)
test_image_pitchs_nor = Normalization(test_image_pitchs)

# 数据展平
train_image_pitchs_nor_spread = train_image_pitchs_nor.reshape(train_image_pitchs_nor.shape[0], -1)
print(train_image_pitchs_nor_spread.shape,'asdasdasdasdasdasdasdad')
train_image_pitchs_second_nor_spread = train_image_pitchs_second_nor.reshape(train_image_pitchs_second_nor.shape[0], -1)
test_image_pitchs_nor_spread = test_image_pitchs_nor.reshape(test_image_pitchs_nor.shape[0], -1)

# # 调用PCA进行降维，降为20维，降维前要把数据展平
# train_image_pitchs = train_image_pitchs.reshape(train_image_pitchs.shape[0], -1)
# test_image_pitchs = test_image_pitchs.reshape(test_image_pitchs.shape[0], -1)
# train_image_pitchs_pca, train_image_pitchs = pca(train_image_pitchs, 50)
# test_image_pitchs_pca, test_image_pitchs = pca(test_image_pitchs, 50)
# print(test_image_pitchs_pca.shape,'asdasdasdasda')

# 调用K-mean聚类对降维后的数据进行分类,这里聚为200类
K = KMeans(500, 'k-means++', verbose=1000)
K.fit(train_image_pitchs_nor_spread)
print(K.cluster_centers_)
print(K.cluster_centers_.shape,'5465s45fds65dfs')


# 把聚类后的聚类中心与各个数据进行求距离差，每个样本得出10个（聚类中心的数目）特征值，
# 作为新特征
def distance(image1, image2):
    result = np.zeros(image1.shape[0])
    for i in range(image1.shape[0]):
        a = (np.array(image1[i] - image2[i])) ** 2
        b = np.sum(a)
        result[i] = b
    return result


def change_feat(X, center_num, center,pitch_num):
    '''

    :param X:处理过的切片数据集
    :param center_num: 聚类层数
    :param center: 聚类中心的矩阵（center_num,feature_num）
    :param pitch_num: 一张图切成多少块
    :return: X_return的数量为切片数据集数量除以一张图的切片数,维度为4*聚类层数
    '''
    X_return = np.zeros((int(X.shape[0]/pitch_num), int(pitch_num/9*center_num)))  # 初始值为（N,pitch_num/4*center_num）
    k = 0
    layer = np.zeros((center_num, pitch_num))
    for i in range(X.shape[0]):  # 对于每一个样本的一组特征值来说
        a = np.tile(X[i].reshape(-1, 1), (1, center_num)).T  # 将X[i]从（1,D）复制为（D,C）
        num = distance(a, center)
        c = np.argmax(num)
        layer[c][k] = 1
        k += 1
        if k == 36:             #每遍历完一整张图切成的块，36个，重新构造特征多层矩阵
            b = int(np.sqrt(pitch_num)/2)
            layer = layer.reshape(center_num,int(np.sqrt(pitch_num)),int(np.sqrt(pitch_num)))
            images_1 = np.mean(layer[:,:b,:b],axis=(1,2))#池化层 分割 池化
            images_2 = np.mean(layer[:,:b,b:],axis=(1,2))
            images_3 = np.mean(layer[:,b:,:b],axis=(1,2))
            images_4 = np.mean(layer[:,b:,b:],axis=(1,2))
              #合并池化层
            f = np.zeros(int(center_num * 4))
            for b in range(center_num):         #把每一层的值输入到特征中
                f[b*4],f[1+b*4],f[2+b*4],f[3+b*4]=images_1[b],images_2[b],images_3[b],images_4[b]
            X_return[int(i/36)] = f.T
            k = 0
            layer = np.zeros((center_num, pitch_num))
    return X_return


best_train = change_feat(train_image_pitchs_second_nor,K.cluster_centers_.shape[0], K.cluster_centers_,36)
print(best_train.shape, 'asdasdasda', best_train[1])
best_test = change_feat(test_image_pitchs_nor_spread, K.cluster_centers_.shape[0], K.cluster_centers_,36)
model = SVC(gamma='scale',kernel='rbf')
model.fit(best_train, train_labels)

print('test 正确率%f' % (np.mean(model.predict(best_test) == test_labels)))
