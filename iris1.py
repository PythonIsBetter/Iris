import numpy as np
from matplotlib import colors
from sklearn import svm
from sklearn.svm import SVC
from sklearn import model_selection
import matplotlib.pyplot as plt
import matplotlib as mpl


# 数据准备
# データ準備

# 在函数中建立一个对应字典就可以了，输入字符串，输出字符串对应的数字。
# 英文字の名称を数字にする
def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


# データ読込
data_path = './iris.data'  # 数据文件的路径　ファイルパス
data = np.loadtxt(data_path,  # 数据文件路径 ファイルパス
                  dtype=float,  # 数据类型　タイプを交換
                  delimiter=',',  # 数据分隔符　コンマ区切り
                  converters={4: iris_type})  # 将第5列使用函数iris_type进行转换　上のiris_typeで5列目のデータを数字にする
print(data)  # data为二维数组，data.shape=(150, 5) 　dataは二次元配列

# 数据分割
# データを切る　前の4列はデータの特徴、最後の1列はデータの結果

x, y = np.split(data,  # 要切分的数组
                (4,),  # 沿轴切分的位置，第5列开始往后为y
                axis=1)  # 1代表纵向分割，按列分割

x = x[:, 0:2]  # 用花萼的长度和宽度进行训练 萼の長さ、幅を使って訓練する
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,  # 所要划分的样本特征集 特徴セット
                                                                    y,  # 所要划分的样本结果　結果セット
                                                                    random_state=1,  # 随机数种子确保产生的随机数组相同
                                                                    test_size=0.3)  # 测试样本占比　30%のデータをテストする、70%のを訓練する


# 模型搭建
# SVMモジュール構築

def classifier():
    clf = svm.SVC(C=0.5,  # 误差项惩罚系数,默认值是1 精度を定義する、1はとてもゆとり、0はとても厳しい
                  kernel='linear',  # 线性核 線形回帰の手法を使う
                  decision_function_shape='ovr')  # 决策函数 決定木を確定
    return clf


clf = classifier()  # 实例化　インスタンス作り


# 模型训练
# モジュール訓練

def train(clf, x_train, y_train):
    clf.fit(x_train,  # 训练集特征向量，fit表示输入数据开始拟合
            y_train.ravel())  # 训练集目标值 ravel()扁平化，将原来的二维数组转换为一维数组


train(clf, x_train, y_train)  # 用训练集进行训练


# 模型评估
# モジュール評価

# 判断a b是否相等，计算acc的均值
# accの平均値を計算
def show_accuracy(a, b, tip):
    acc = a.ravel() == b.ravel()
    print('%s Accuracy:%.3f' % (tip, np.mean(acc)))


# 分别打印训练集和测试集的准确率
# 訓練セットとテストセットの精度を表示
def print_accuracy(clf, x_train, y_train, x_test, y_test):
    # 分别打印训练集和测试集的准确率  score(x_train,y_train):表示输出x_train,y_train在模型上的准确率
    print('training prediction:%.3f' % (clf.score(x_train, y_train)))  # 訓練セットの精度を表示
    print('test data prediction:%.3f' % (clf.score(x_test, y_test)))  # テストセットの精度を表示
    # 原始结果与预测结果进行对比   predict()表示对x_train样本进行预测，返回样本类别
    show_accuracy(clf.predict(x_train), y_train, 'training data')  # 訓練セットの特徴データを用いて、結果を予測する。予測結果と真実結果を比較する
    show_accuracy(clf.predict(x_test), y_test, 'testing data')  # テストセットの特徴データを用いて、結果を予測する。予測結果と真実結果を比較する
    # 计算决策函数的值，表示x到各分割平面的距离,3类，所以有3个决策函数，不同的多类情况有不同的决策函数？
    print('decision_function:\n', clf.decision_function(x_train))  # 決定木の距離（ステップ数）


print_accuracy(clf, x_train, y_train, x_test, y_test)


def draw(clf, x):
    iris_feature = 'sepal length', 'sepal width', 'petal lenght', 'petal width'
    # 开始画图
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点 开始坐标：结束坐标（不包括）：步长
    # flat将二维数组转换成1个1维的迭代器，然后把x1和x2的所有可能值给匹配成为样本点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # stack():沿着新的轴加入一系列数组，竖着（按列）增加两个数组，grid_test的shape：(40000, 2)
    print('grid_test:\n', grid_test)
    # 输出样本到决策面的距离
    z = clf.decision_function(grid_test)
    print('the distance to decision plane:\n', z)

    grid_hat = clf.predict(grid_test)  # 预测分类值 得到【0,0.。。。2,2,2】
    print('grid_hat:\n', grid_hat)
    grid_hat = grid_hat.reshape(x1.shape)  # reshape grid_hat和x1形状一致
    # 若3*3矩阵e，则e.shape()为3*3,表示3行3列
    # light是网格测试点的配色，相当于背景
    # dark是样本点的配色
    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'b', 'r'])
    # 画出所有网格样本点被判断为的分类，作为背景
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # pcolormesh(x,y,z,cmap)这里参数代入
    # x1，x2，grid_hat，cmap=cm_light绘制的是背景。
    # squeeze()把y的个数为1的维度去掉，也就是变成一维。
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), edgecolor='k', s=50, cmap=cm_dark)  # 样本点
    plt.scatter(x_test[:, 0], x_test[:, 1], s=200, facecolor='yellow', zorder=10, marker='+')  # 测试点
    plt.xlabel(iris_feature[0], fontsize=20)
    plt.ylabel(iris_feature[1], fontsize=20)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title('svm in iris data classification', fontsize=30)
    plt.grid()
    plt.show()


draw(clf, x)
