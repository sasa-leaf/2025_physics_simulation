import taichi as ti
import numpy
import os
ti.init(arch=ti.cpu)

# 計算領域
domain = ti.Vector.field(2,ti.f32,shape=(2))
domain[0] = (-1.0,-1.0) # (x_min,y_min)
domain[1] = (1.0,1.0) # (x_max,y_max)

# 質点の数
N_points = 1

# パラメータ
points_mass = ti.field(ti.f32,shape=(N_points))
points_radius = ti.field(ti.f32,shape=(N_points))

# 変数
points_pos = ti.Vector.field(2,ti.f32,shape=(N_points))
points_vel = ti.Vector.field(2,ti.f32,shape=(N_points))
points_acc = ti.Vector.field(2,ti.f32,shape=(N_points))
points_force = ti.Vector.field(2,ti.f32,shape=(N_points))

# 時間
step = ti.field(ti.i32,shape=())
time = ti.field(ti.f32,shape=())
dt = 0.01

# 初期化カーネル
@ti.kernel
def initialize():
  for i in range(N_points):
    # パラメータ
    points_mass[i] = 1.0
    points_radius[i] = 0.1
    # 初期位置
    points_pos[i][0] = -0.5
    points_pos[i][1] = 0.5
    # 初期速度
    points_vel[i][0] = 1.0
    points_vel[i][1] = 0.0
  step[None] = 0
  time[None] = 0.0

# 更新カーネル
@ti.kernel
def update():
  # 力の計算
  for i in range(N_points):
    points_force[i].fill(0.0)
  # 加速度の計算
  for i in range(N_points):
    points_acc[i] = points_force[i]/points_mass[i]
  # 速度の更新
  for i in range(N_points):
    points_vel[i] += points_acc[i]*dt
  # 位置の更新
  for i in range(N_points):
    points_pos[i] += points_vel[i]*dt
  # 時間ステップの更新
  step[None] += 1
  time[None] = step[None]*dt

# 初期化する
initialize()

window_size = (640,640)
scale_to_pixel = window_size[0]/(domain[1]-domain[0]).x
gui = ti.GUI(os.path.basename(__file__),window_size)

while gui.running:
  # 時間を進める
  update()
  # キーボードとマウスクリックの受け取り
  for e in gui.get_events(gui.PRESS):
    # ESCキーが押されたら初期化する
    if e.key == ti.GUI.ESCAPE:
      initialize()
  # 現在の状態を描画する
  X = points_pos.to_numpy()
  X[:,0] = (X[:,0]-domain[0].x)/(domain[1]-domain[0]).x
  X[:,1] = (X[:,1]-domain[0].y)/(domain[1]-domain[0]).y
  R = points_radius.to_numpy()
  V = points_vel.to_numpy()*0.1
  A = points_acc.to_numpy()*0.01
  gui.circles(X,radius=R*scale_to_pixel,color=0x808080)
  gui.arrows(X,V,radius=2,color=0xFFFFFF) # 速度
  gui.arrows(X,A,radius=2,color=0xFF00FF) # 加速度
  gui.text(f'Step: {step[None]}, Time: {time[None]:.6f}',(0.0,1.0),font_size=20,color=0xFFFFFF)
  gui.show()
  