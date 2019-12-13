#凝固についての気圧と温度

from sklearn.metrics import accuracy_score

# アルゴリズム
from sklearn.svm       import LinearSVC

import warnings
warnings.filterwarnings('ignore')

# 学習データ　[温度,気圧]
in_data = [
    [100, 1],
    [ 60, 1],
    [  2, 1],
    [ 40, 1],
    [  1, 1],
    [ 20, 1],
    [0.01,1],
    [  3, 1],
    [ -1, 1],
    [-20, 1],
    [ -2, 1],
    [-0.3,1]
]

# 学習データ
out_data = ['水','水','水','水','水','水','水','水','氷','氷','氷','氷']

#予測結果
result_data = ['水', '水', '氷', '水']

# アルゴリズムの設定
clf = LinearSVC()

# 学習
clf.fit(in_data, out_data)

# テストデータ(予測したいデータ))
test_data = [
    [ 50, 1],
    [ 70, 1],
    [ -4, 1],
    [ 30, 1]
]

# 予測
result = clf.predict(test_data)
print("正解:", out_data)
print("予測結果:", result)

print("正解率 = " , accuracy_score(result_data, result))
