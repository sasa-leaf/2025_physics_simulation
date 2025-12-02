import taichi as ti
import numpy

# リストからベクトルを作成
a = ti.Vector([1,2,3])

# numpy配列からベクトルを作成
b = ti.Vector(numpy.array([4,5,6]))

# リストから行列を作成
m = ti.Matrix([[1,1,2],[0,1,0],[0,0,1]])

# ベクトルの和
x = a+b
print(x,type(x))

# ベクトルの内積
x = a.dot(b)
print(x,type(x))

# 行列ベクトル積
x = m@a
print(x,type(x))

# 行列行列積
x = m@m
print(x,type(x))

# 演算結果をnumpyへ変換
x = (a+m@b).to_numpy()
print(x,type(x))
