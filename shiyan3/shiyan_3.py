from sklearn import svm
from sklearn.cluster import KMeans
import numpy as np
import gzip
import pickle
def load_data(train_batch_size, test_batch_size):
    '''加载数据集'''   
    f = gzip.open('F:/jupyter/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    train_data = training_data[0]
    train_label = training_data[1]
    train_data = train_data[:train_batch_size]
    train_label = train_label[:train_batch_size]

    test_data = test_data[0][:test_batch_size]
    test_label = test_data[1][:test_batch_size]
    return train_data, train_label, test_data, test_label
def normalize(data, threshold):
        """图片像素二值化，变成0-1分布"""
        for i in range(len(data)):
            for j in range(len(data[0])):
                if data[i][j] > threshold:
                    data[i][j] = 1
                else:
                    data[i][j] = 0
        return data
def pca(data, top_n_feature):
        '''用PCA对数据矩阵降维'''
        mean_vals = data.mean(axis = 0)
        changed_data = data - mean_vals
        cov_mat = np.cov(changed_data, rowvar=0)
        feature_vals, all_feature_vects = np.linalg.eig(np.mat(cov_mat))
        small_to_large_index = np.argsort(feature_vals)
        large_to_small_index = small_to_large_index[::-1]
        index_expected = large_to_small_index[:top_n_feature]
        feature_vects = all_feature_vects[:, index_expected]
        feature_vects = np.real(feature_vects)
        low_dim_data = changed_data.dot(feature_vects)
        restructure_data = low_dim_data.dot(feature_vects.T) + mean_vals 
        return restructure_data
def cal_accuracy(pred_test, test_labels):
        '''计算预测正确率'''
        num_correct = 0 
        for a, y in zip(pred_test, test_labels):
            if a == y:
                num_correct += 1
        return 1.0 * num_correct / len(test_labels)
class A():
    '''用来复现论文的类'''
    def __init__(self, random_blocks_num_per_img, block_size, k, s):
        ''' train_data :表示训练数据输入
            test_data ：测试数据
            random_blocks_num_per_img ：每张图随机取块个数
            block_size ：随机取块和固定位置取块时的块边长
            k：聚类中心个数 
            s ：固定取块时的步长，因为图片大小确定，所以固定取块个数由步长决定
        '''
        self.random_blocks_num_per_img = random_blocks_num_per_img
        self.block_size = block_size
        self.k = k
        self.s = s
        '''一定要把图片坐标轴想象成正常的，倒过来的容易想不明白'''
    def get_random_blocks(self, data):
        '''用来随机取块'''
        m = len(data)
        n = int(np.sqrt(len(data[0])))
        blocks_random = np.zeros((m * self.random_blocks_num_per_img, self.block_size**2 ))
        for i, img in enumerate(data):
            img = img.reshape(n, n)
            for j in range(self.random_blocks_num_per_img):
                x = np.random.randint(0, n - self.block_size - 1)
                y = np.random.randint(0, n - self.block_size - 1)
                #block_img = img[range(x, x + self.block_size), range(y, y + self.block_size)]
                block_img = img[x:x + self.block_size, y:y + self.block_size]
                block_img = block_img.reshape(1, self.block_size**2)
                blocks_random[i * self.random_blocks_num_per_img + j] = block_img
        return blocks_random
    def get_static_blocks(self, data):
        '''用来获取固定取块'''
        m = int(len(data))
        n = int(np.sqrt(len(data[0])))
        x_num = int((n - self.block_size) // self.s)
        y_num = int((n - self.block_size) // self.s)
        if x_num % 2 == 1:
            x_num = int(x_num - 1)
            y_num = int(y_num - 1)
        blocks_static = np.zeros((m * x_num * y_num, self.block_size**2))
        for i, img in enumerate(data):
            img = img.reshape(n, n)
            for y in range(y_num):
                for x in range(x_num):
                    x1 = x * self.s
                    x2 = x1 + self.block_size
                    y1 = y * self.s
                    y2 = y1 + self.block_size
                    block_img = img[x1:x2, y1:y2]
                    block_img = block_img.reshape(1, self.block_size**2)
                    blocks_static[i * x_num * y_num + y * x_num + x] = block_img
        return blocks_static, x_num, y_num
    def distance(self, img1, img2):
        length = img1.shape[0]
        result = np.zeros(length)
        for i in range(length):
            temp = np.sum(np.array(img1[i] - img2[i]) ** 2)
            result[i] = temp
        return result

    def cal_vects(self, data, blocks, centers, x_num, y_num):
        """计算图片的词向量矩阵"""
        vects = np.zeros((len(data), self.k))
        t = x_num * y_num
        for i in range(len(data)):
            for j in range(t):

                index = 0
                min = np.sum((blocks[i*t+j] - centers[0])**2)
                for k in range(1, len(centers)):
                    if np.sum((blocks[i*t+j] - centers[k])**2) < min:
                        index = k
                        min = np.sum((blocks[i*t+j] - centers[k])**2)

                vects[i][index] += 1
        return vects
    def k_means(self, data, points, mat, labels, m, n, k):
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
    def get_vects(self, data, blocks, centers, x_num, y_num):
        '''求各个图片的图像向量'''
        t = x_num * y_num
        middle = int(x_num / 2)
        data_length = len(data)
        blocks_length = len(blocks)
        expected_vect = np.zeros((data_length, 4 * self.k))
        vect_1 = np.zeros((x_num, y_num, self.k))
        for i in range(blocks_length):
            each_block = np.tile(blocks[i].reshape(1, -1), (self.k, 1))
            num = self.distance(each_block, centers)
            index = np.argsort(num)
            if (i + 1) % t != 0:
                vect_1[((i % t) % x_num)][((i % t) // x_num)][index[0]] = 1
            else:
                part_1 = np.max(vect_1[:middle, :middle, :], axis=(0, 1))
                part_2 = np.max(vect_1[middle:x_num, :middle, :], axis=(0, 1))
                part_3 = np.max(vect_1[:middle, middle:y_num, :], axis=(0, 1))
                part_4 = np.max(vect_1[middle:x_num, middle:y_num, :], axis=(0, 1))

                vect_2 = np.zeros(int(4 * self.k))
                for j in range(self.k):
                    vect_2[4*j],vect_2[4*j+1],vect_2[4*j+2],vect_2[4*j+3] = part_1[j],part_2[j],part_3[j],part_4[j]
                expected_vect[(i+1)//t - 1] = vect_2
                vect_1 = np.zeros((x_num, y_num, self.k))          
        return expected_vect

train_data, train_label, test_data, test_label = load_data(3000, 400)
m =10*3000
train_data = normalize(train_data, threshold = 0)
test_data = normalize(test_data, threshold = 0)
print('在1这里')
#top_n_feature = 1
#train_data = pca(train_data, top_n_feature)
#test_data = pca(test_data, top_n_feature)
# 胖类和瘦类
# k表示聚类中心个数
cluster = 100
each_block_size = 10
stride = 3
f = A(random_blocks_num_per_img=10, block_size=each_block_size, k=cluster, s=stride)

random_blocks = f.get_random_blocks(train_data)
print('在2这里')
    
centers = np.random.rand(cluster, each_block_size**2)
mat = np.zeros((m, cluster))
labels = np.zeros(m)
times = 0
if times < 30:
    centers, mat = f.k_means(random_blocks, centers, mat, labels, m, each_block_size**2, cluster)
    times += 1
print('在3这里')

#points = KMeans(n_clusters=cluster, max_iter=300).fit(random_blocks)
#centers = points.cluster_centers_
#print('在3这里')

train_static_blocks, x_num, y_num= f.get_static_blocks(train_data)
print('在4这里')
test_static_blocks, x_num, y_num= f.get_static_blocks(test_data)
print('在5这里')

train_vect = f.cal_vects(train_data, train_static_blocks, centers, x_num, y_num)
print('在6这里')
test_vect = f.cal_vects(test_data, test_static_blocks, centers, x_num, y_num)
print('在7这里')

model = svm.SVC(kernel='rbf', max_iter=300)
model.fit(train_vect, train_label)
print('在8这里')

pred_test = [int(a) for a in model.predict(test_vect)]
print('在9这里')

accuracy = cal_accuracy(pred_test, test_label)
print(accuracy)