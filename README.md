## 目次
* [「アヤメ品種分類」とは](#「アヤメ品種分類」とは)
* [iris.data](#iris.data)
* [iris1.py](#iris1.py)
* [iris2.py](#iris2.py)

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
    
    他の30%のデータ（45個）をテスト/評価データとして使う。
    
    SVMモジュールを構築して訓練させ、テストデータでこのモジュールの精度を評価した。
### 出力
#### 全てのデータを表示する
[](/img/1.PNG)
