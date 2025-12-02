import taichi as ti
import numpy
import os
ti.init(arch=ti.cpu)

# 点の数
N_points = 500

# 位置を格納する配列
points_pos = ti.Vector.field(2,ti.f32,shape=(N_points))

# 初期化カーネル
@ti.kernel
def initialize():
  for i in range(N_points):
    points_pos[i][0] = ti.random() # 0〜1の乱数
    points_pos[i][1] = ti.random() # 0〜1の乱数

# 初期化を実行
initialize()

# GUIを作成
window_title = os.path.basename(__file__) # ファイル名をウィンドウタイトルに設定
window_size = (640,640)
gui = ti.GUI(window_title, window_size)

while gui.running:
  # 描画
  gui.circles(points_pos.to_numpy(),radius=10,color=0x0000FF)
  gui.show()