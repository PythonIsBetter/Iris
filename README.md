## 目次
* [「アヤメ品種分類」とは](#アヤメ品種分類とは)
* [iris.data](#irisdata)
* [iris1.py](#iris1py)
* [iris2.py](#iris2py)

「アヤメ品種分類」とは
------
    アヤメという花の4つの属性（がく片の長さ、がく片の幅、花びらの長さ、花びらの幅）より、3種類を分類することができる。
    
    この中に、「回帰」と「分類」という機械学習の基本手法を実践できる。
    
    このプロジェクトでは、線形回帰を中心として、2つの方法で分類を実現した。
    
iris.data
------
    アヤメのデータを含まれているファイルである。
    
    全部150個のセットがある、1個のセットに、がく片の長さ、がく片の幅、花びらの長さ、花びらの幅、種類を含まれている。
    
    例えば、[5.1,3.5,1.4,0.2,Iris-setosa]このセットに、
    
    がく片の長さは5.1cm、がく片の幅は3.5cm、花びらの長さは1.4cm、花びらの幅は0.2cm、種類はIris-setosaである。
    
    
    今回はこれらのデータを使って、モジュールを訓練して、精度を評価したり、未知のデータを予測したりする。

iris1.py
------
### 解説
    ローカルからアヤメのデータをプログラムに読み込んで、
    
    70%のデータ（105個）を訓練用のデータとして使って、
    
    他の30%のデータ（45個）をテストデータとして使う。
    
    1個のセットに、前の4列は特徴量を、最後の1列を分類結果として使う。
    
    決定木というアルゴリズムのSVMモジュールを構築して訓練させられて、テストデータでこのモジュールの精度を評価した。
### 出力
#### ファイル中の全てのデータを表示する
（キャプチャは一部のデータ）\
![](/img/1.PNG)

#### モジュールの精度
    精度が0の場合は、データが無相関
    
    精度が1の場合は、データの当てはまりがよい
    

    1行目：訓練データをモジュールに入れて、精度を評価する
    
    2行目：テストデータをモジュールに入れて、精度を評価する
    
    3行目：訓練データの特徴量をモジュールに入れて分類結果を予測する。予測の結果と真実の結果を比較して、精度を求める
    
    4行目：テストデータの特徴量をモジュールに入れて分類結果を予測する。予測の結果と真実の結果を比較して、精度を求める
![](/img/2.PNG)

#### 決定木の距離
    それぞれの特徴量に対して、必ず3つの目的地（アヤメの種類）の一つにたどり着く\
    
    値が大きければ大きいほど、真実の種類に近い

（キャプチャは一部のデータ）\
![](/img/3.PNG)

iris2.py
------
### 解説
    この方法で、k近傍法を使ってモジュールを構築する。
    
    sklearnというライブラリに、アヤメのデータはもう既に含まれているので、今回データの読込はなし。
    
    まずは4つの特徴量から散布図を描く。散布図で、3種類のアヤメの特徴の関係がとても見やすくなる。
    
    次はk近傍法でモジュールを構築して訓練させられる。
    
    最後は任意の特徴量をモジュールに与えて予測させて、モジュールの精度を評価する
    
    
### 出力
#### sklearnから取り出したデータのキー
![](/img/4.PNG)

#### sklearnでのアヤメデータについての説明
![](/img/5.PNG)

#### 3種類と4特徴量
![](/img/6.PNG)

#### 特徴量セットの前の3列
![](/img/7.PNG)


#### 特徴量セット大きさとタイプ
![](/img/8.PNG)


#### 3種類のデータセット
![](/img/9.PNG)

#### 訓練データとテストの大きさとタイプ
![](/img/10.PNG)

#### 散布図
    4つの特徴の間に、それぞれの線形関係を表示する
![](/img/11.PNG)

#### 任意の特徴量から予測できた結果とこの結果の名称
![](/img/12.PNG)


#### k近傍法モジュールの精度
![](/img/13.PNG)
