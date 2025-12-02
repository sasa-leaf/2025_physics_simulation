import taichi as ti
import numpy
import os
ti.init(arch=ti.cpu)

# 計算領域
domain = ti.Vector.field(2,ti.f32,shape=(2))
domain[0] = (-2.0,-1.0)
domain[1] = (2.0,1.0)

# 点の数
distance = 0.2
N_points_x = int((domain[1]-domain[0]).x/distance+0.5)
N_points_y = int((domain[1]-domain[0]).y/distance+0.5)
N_points = N_points_x * N_points_y
print("Number of points:",N_points)

# 位置ベクトル
points_pos = ti.Vector.field(2,ti.f32,shape=(N_points))

# 初期化カーネル
@ti.kernel
def initialize():
  for i in range(N_points):
    points_pos[i][0] = domain[0].x+distance*(i % N_points_x+0.5)
    points_pos[i][1] = domain[0].y+distance*(i // N_points_x+0.5)

    # ウィンドウ座標に変換 (0,0)が左下，(1,1)が右上
    points_pos[i][0] = (points_pos[i][0]-domain[0].x)/(domain[1].x-domain[0].x)
    points_pos[i][1] = (points_pos[i][1]-domain[0].y)/(domain[1].y-domain[0].y)
    print(points_pos[i])


# 初期化を実行
initialize()

# GUIを作成
window_title = os.path.basename(__file__) # ファイル名をウィンドウタイトルに設定
window_size = (640,320)
gui = ti.GUI(window_title, window_size)

while gui.running:
  # 描画
  gui.circles(points_pos.to_numpy(),radius=640//N_points_x//2,color=0x0000FF)
  gui.show()