from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 测试数据
# データを読込、sklearnというライブラリにデータセットを含まれている
iris_dataset = load_iris()  # 加载数据，返回一个Bunch对象，类似字典
print("Keys of iris_dataset:\n{}", format(iris_dataset.keys()))  # 键值
print(iris_dataset['DESCR'][:193] + '\n...')  # 包含数据集简要说明
print(iris_dataset['target_names'])  # 字符串数组，预测的目标
print(iris_dataset['feature_names'])  # 字符串列表，样本属性
print(iris_dataset['data'][:3])  # 输入数据，NumPy二维数组 【样本】【属性】
print(iris_dataset['data'].shape)
print(type(iris_dataset['data']))
print(iris_dataset['target'])  # 目标，NumPy一维数组

# 衡量模型
# データを切る、全てのデータを訓練データとして使う
x_train, x_test, y_train, y_test = train_test_split(  # 打乱数据集并拆分成训练数据和测试数据
    iris_dataset['data'],
    iris_dataset['target'],
    random_state=0)  # 指定随机数生成种子，保证多次运行同一函数有相同的输出
print(x_train.shape)
print(x_test.shape)

# 观察数据，散点图
# 散布図を描く
iris_dataframe = pd.DataFrame(x_train, columns=iris_dataset.feature_names)  # 创建散点图矩阵　描くデータフレームを作る
grr = pd.plotting.scatter_matrix(
    iris_dataframe,
    c=y_train,
    figsize=(15, 15),
    marker='o',
    hist_kwds={'bins': 20},
    s=60,
    alpha=.8,
    cmap=mglearn.cm3)  # 绘制散点图 図を描く
plt.show()

# 构建模型：K近邻算法
# k近傍法で訓練する
knn = KNeighborsClassifier(n_neighbors=1)  # 创建一个实例，封装了相关算法
knn.fit(x_train, y_train)  # 基于训练集构建模型，返回对象本身并做出修改

# 做出预测
# データを予測する
x_new = np.array([[5, 2.9, 1, 0.2]])  # 二维数组 指定されたデータ
prediction = knn.predict(x_new)  # 予測する
print(prediction)  # 予測結果を表示する
print(iris_dataset['target_names'][prediction])  # 予測結果の名称を表示する

# 评估模型
# モジュールの精度を評価する
y_pred = knn.predict(x_test)
print(np.mean(y_pred == y_test))
