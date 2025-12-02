import taichi as ti
import numpy
import os
ti.init(arch=ti.cpu)

# 計算領域
domain = ti.Vector.field(2,ti.f32,shape=(2))
domain[0] = (-1.0,-1.0)
domain[1] = (1.0,1.0)

# 点の数と半径
N_points = 3
radius = 0.1

# 位置と速度
points_pos = ti.Vector.field(2,ti.f32,shape=(N_points))
points_vel = ti.Vector.field(2,ti.f32,shape=(N_points))

# 時間
time = ti.field(ti.f32,shape=())
dt = 0.005 # 時間ステップ幅

# 初期化
@ti.kernel
def initialize():
    points_pos[0] = (-1.0,-0.5)
    points_vel[0] = (1.0,0.5)
    points_pos[1] = (0.5,-1.0)
    points_vel[1] = (-0.5,1.0)
    points_pos[2] = (1.0,0.5)
    points_vel[2] = (-1.0,-0.5)
    time[None] = 0.0

# 時間をdtだけ進める
@ti.kernel
def update():
    for i in range(N_points):
        points_pos[i] += points_vel[i]*dt
    time[None] += dt

# 初期化を実行
initialize()

# GUIを作成
window_title = os.path.basename(__file__) # ファイル名をウィンドウタイトルに設定
window_size = (640,640)
gui = ti.GUI(window_title, window_size)

while gui.running:
    # 時間発展
    update()

    X = points_pos.to_numpy()
    for i in range(N_points):
        X[i][0] = (X[i][0]-domain[0].x)/(domain[1].x-domain[0].x)
        X[i][1] = (X[i][1]-domain[0].y)/(domain[1].y-domain[0].y)

    # 描画
    gui.circles(X, radius=32, color=0x0000FF)
    
    start_pos = X
    print("Draw velocity arrow for point",i,"from",start_pos)
    gui.arrows(start_pos, points_vel.to_numpy()*0.1, radius=4, color=0xFF0000)

    gui.show()