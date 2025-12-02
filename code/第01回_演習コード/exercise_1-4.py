import taichi as ti
import numpy as np
import os
ti.init(arch=ti.cpu)

"""
粒子同士の衝突判定を追加したバージョン．

1. x方向の衝突を定義する
2. y方向の衝突を定義する
3. 衝突時の位置関係から，速度の更新を行う

n: iからjへの単位法線ベクトル
n = (x_j - x_i) / ||x_j - x_i||
u_i_new = u_i + n * ( (u_j - u_i) · n )
u_j_new = u_j - n * ( (u_j - u_i) · n )
"""

# 計算領域
domain = ti.Vector.field(2,ti.f32,shape=(2))
domain[0] = (-1.0,-1.0)
domain[1] = (1.0,1.0)

# 点の数と半径
N_points = 20
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
        if p.x < (minx + radius) and v.x < 0:
            v.x = -v.x
        elif p.x > (maxx - radius) and v.x > 0:
            v.x = -v.x

        # y方向の壁衝突
        if p.y < (miny + radius) and v.y < 0:
            v.y = -v.y
        elif p.y > (maxy - radius) and v.y > 0:
            v.y = -v.y

        # 粒子間衝突
        for j in range(N_points):
            if i != j:
                # 距離を計算
                dir_ij = points_pos[j] - p
                dist_ij = dir_ij.norm()

                if dist_ij < 2.0 * radius and (points_vel[j] - v).dot(dir_ij) < 0:
                    n_ij = dir_ij / dist_ij # 単位法線ベクトル
                    relative_velocity = points_vel[j] - v # 相対速度
                    
                    # 式に基づいて速度更新
                    v += n_ij * (relative_velocity.dot(n_ij))
                    points_vel[j] -= n_ij * (relative_velocity.dot(n_ij))

        points_pos[i] = p
        points_vel[i] = v
    time[None] += dt

# 初期化を実行
initialize()

# GUIを作成
window_title = os.path.basename(__file__)
window_size = (640,640)
gui = ti.GUI(window_title, window_size)

while gui.running:
    # 時間発展
    update()

    # 運動エネルギーを計算
    K = 0.0
    for i in range(N_points):
        v = points_vel[i]
        K += 1 * (v.x**2 + v.y**2)
    print("時刻:", time[None], "運動エネルギー:", K)

    X = points_pos.to_numpy()
    for i in range(N_points):
        X[i][0] = (X[i][0]-domain[0].x)/(domain[1].x-domain[0].x)
        X[i][1] = (X[i][1]-domain[0].y)/(domain[1].y-domain[0].y)

    # 描画
    gui.circles(X, radius=32, color=0x0000FF)
    
    start_pos = X
    # print("Draw velocity arrow for point",i,"from",start_pos)
    gui.arrows(start_pos, points_vel.to_numpy()*0.1, radius=4, color=0xFF0000)

    gui.text(content=f"Time: {time[None]}",pos=[0.01,0.95],font_size=20,color=0xFFFFFF)
    gui.text(content=f", Undo Energy: {K}",pos=[0.4,0.95],font_size=20,color=0xFFFFFF)
    gui.show()