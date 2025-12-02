import taichi as ti
import numpy
ti.init(arch=ti.cpu)

# 単一のスカラー
f0 = ti.field(ti.f32,shape=())
f0[None] = 0.9 # 代入

# スカラーの1次元配列
f1 = ti.field(ti.f32,shape=5)
f1[1] = 1.5 # 代入

# スカラーの2次元配列
f2 = ti.field(ti.f32,shape=(5,5))
f2.fill(0.1) # すべての要素に0.1を代入

# 3Dベクトルの2次元配列
vf = ti.Vector.field(n=3,dtype=ti.f32,shape=(5,5))
vf[0,0] = [1.0,2.0,3.0] # 代入
vf[1,0] = (4.0,5.0,6.0) # 代入

# 3行x2列の行列の2次元配列
mf = ti.Matrix.field(n=3,m=2,dtype=ti.f32,shape=(5,5))
mf[0,0] = [[1.0,2.0],[3.0,4.0],[5.0,6.0]] # 代入

@ti.kernel
def test():
  for i in ti.grouped(mf):
    for j in range(3):
      for k in range(2):
        mf[i][j,k] = ti.random() # 乱数を代入

test()
