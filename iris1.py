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
