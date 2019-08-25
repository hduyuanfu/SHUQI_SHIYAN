from hanshu import *
from sklearn import svm

# 加载数据
training_data, validation_data, test_data_ = load_data()
#m0表示训练集样本数，m1表示对训练集进行随机和固定取块时的取块总数，m2表示测试集样本个数，
# m3表示测试集固定位置取块的总数
m0 = 45000
m1 = 10 * m0
m2 =3000
m3 = 10 * m2

# 获取需要的train_data和labels
train_data = training_data[0]
train_labels = training_data[1]
train_data = train_data[:m0]
train_labels = train_labels[:m0]

# 获取需要的test_data和labels
test_data = test_data_[0]
test_labels = test_data_[1]
test_data = test_data[:m2]
test_labels = test_labels[:m2]

# 如果需要，可以对训练数据进行PCA降维，使图形规整
# train_data = pca(train_data, 1)

# 将数据集二值化
train_data = normalize(train_data)
test_data = normalize(test_data)

# 对训练数据随机取块
block_train_random = blocks(train_data, 10, m1, 0)
print('在这里-1')

# 对随机生成的块进行聚类，k表示个聚类中心个数，n表示截取块的大小（为正方形）
k = 100 
n = 100
# 初始化聚类中心；mat,labels作为k_means函数执行过程中需要的变量       
points = np.random.rand(k, 100)
mat = np.zeros((m1, k))
labels = np.zeros(m1)
# k-means迭代30次
times = 0
if times < 30:
    points, mat = k_means(block_train_random, points, mat, labels, m1, n, k)
    times += 1
print('在这里0')

# 生成训练和测试数据的固定块
block_train_static = blocks(train_data, 10, m1, 1)
print('在这里1')
block_test_static = blocks(test_data, 10, m3, 1)
print('在这里2')

# 生成训练和测试数据的图向量（概念上和词向量很相似，固定截取的块属于聚类模型中的哪一类（可以将聚类得到的
# 各聚类中心默认为0-99等），则该图像的特征向量对应下标（哪一类作为下标）所在位置元素加1）
vects_train = cal_vects(train_data, points, block_train_static)
print('在这里3')
vects_test = cal_vects(test_data, points, block_test_static)
print('在这里4')
# 用svm进行分类模型训练
# 用默认值作为训练模型时需要的参数
moxing = svm.SVC(kernel = 'rbf')
print('在这里5')
# 进行模型训练
moxing.fit(vects_train, train_labels)
print('在这里6')
# 用模型对测试集进行预测
pred_test = [int(a) for a in moxing.predict(vects_test)]
print('在这里7')

# 计算准确率
accuracy = cal_accuracy(pred_test, test_labels)
print(accuracy)
