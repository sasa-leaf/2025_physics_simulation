import taichi as ti
import numpy as np
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
dt = 0.01

@ti.kernel
def initialize():
    for i in range(N_points):
        points_pos[i][0] = ti.random()*2.0-1.0
        points_pos[i][1] = ti.random()*2.0-1.0
        points_vel[i][0] = ti.random()*2.0-1.0
        points_vel[i][1] = ti.random()*2.0-1.0
        points_vel[i] *= 1/ti.math.sqrt(points_vel[i][0]**2+points_vel[i][1]**2)
    time[None] = 0.0

# 位置と速度
points_pos = ti.Vector.field(2,ti.f32,shape=(N_points))
points_vel = ti.Vector.field(2,ti.f32,shape=(N_points))

# 時間をdtだけ進める
@ti.kernel
def update():
    minx, miny = domain[0].x, domain[0].y
    maxx, maxy = domain[1].x, domain[1].y
    for i in range(N_points):
        p = points_pos[i]
        v = points_vel[i]

        # まず移動
        p += v * dt

        # x方向の壁衝突
        if p.x < (minx + radius):
            v.x = -v.x
        elif p.x > (maxx - radius):
            v.x = -v.x

        # y方向の壁衝突
        if p.y < (miny + radius):
            v.y = -v.y
        elif p.y > (maxy - radius):
            v.y = -v.y

        points_pos[i] = p
        points_vel[i] = v
    time[None] += dt

# 初期化を実行
initialize()

# GUIを作成
window_title = os.path.basename(__file__)
window_size = (640,640)
gui = ti.GUI(window_title, window_size)

# 以下のロジックでウィンドウ座標に変換 (0,0)が左下，(1,1)が右上

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