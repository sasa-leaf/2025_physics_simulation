import taichi as ti
import numpy
import os
ti.init(arch=ti.cpu)

# 計算領域
domain = ti.Vector.field(2,ti.f32,shape=(2))
domain[0] = (-1.0,-1.0) # (x_min,y_min)
domain[1] = (1.0,1.0) # (x_max,y_max)

# 重力
gravity = ti.Vector([0.0,-9.8])

# 反発係数
collision_coef = 0.9

# マウスからの力を考慮するための変数
mouse_pos = ti.Vector.field(2,ti.f32,shape=())
mouse_flag = ti.field(ti.i32,shape=())
mouse_radius = 0.2

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
energy = ti.field(ti.f32,shape=())

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
    points_force[i] += points_mass[i]*gravity
    # 下の壁面との衝突
    if abs(domain[0].y-points_pos[i].y) <= points_radius[i] and points_vel[i].y < 0:
      points_force[i].y += -(1.0+collision_coef)*points_mass[i]/dt*points_vel[i].y
      points_force[i].y += -points_mass[i]*gravity.y # 抗力を追加
    # 上の壁面との衝突
    if abs(domain[1].y-points_pos[i].y) <= points_radius[i] and points_vel[i].y > 0:
      points_force[i].y += -(1.0+collision_coef)*points_mass[i]/dt*points_vel[i].y
      points_force[i].y += -points_mass[i]*gravity.y # 抗力を追加
    # 左の壁面との衝突
    if abs(domain[0].x-points_pos[i].x) <= points_radius[i] and points_vel[i].x < 0:
      points_force[i].x += -(1.0+collision_coef)*points_mass[i]/dt*points_vel[i].x
      points_force[i].x += -points_mass[i]*gravity.x # 抗力を追加
    # 右の壁面との衝突
    if abs(domain[1].x-points_pos[i].x) <= points_radius[i] and points_vel[i].x > 0:
      points_force[i].x += -(1.0+collision_coef)*points_mass[i]/dt*points_vel[i].x
      points_force[i].x += -points_mass[i]*gravity.x # 抗力を追加
    # マウスからの力を計算する
    if mouse_flag[None]:
      pos_ij = mouse_pos[None]-points_pos[i]
      dist_ij = pos_ij.norm()
      if dist_ij < mouse_radius+points_radius[i]:
        normal_ij = pos_ij/dist_ij # 単位法線ベクトル
        tmp1 = normal_ij.dot(-points_vel[i]) # 法線方向の相対速度
        if tmp1 < 0.0:
          points_force[i] += normal_ij*(points_mass[i]*tmp1/dt)
        tmp2 = mouse_radius+points_radius[i]-dist_ij # 重なりの長さ
        if tmp2 > 0.0:
          points_force[i] -= normal_ij*(points_mass[i]*tmp2*0.1/dt**2)
  # 加速度の計算
  for i in range(N_points):
    points_acc[i] = points_force[i]/points_mass[i]
  # 速度の更新
  for i in range(N_points):
    points_vel[i] += points_acc[i]*dt
  # 位置の更新
  for i in range(N_points):
    # points_pos[i] += points_vel[i]*dt # 変更前
    points_pos[i] += points_vel[i]*dt-0.5*points_acc[i]*dt**2 # 変更後
  # エネルギー
  energy[None] = 0.0
  for i in range(N_points):
    energy[None] += 0.5*points_mass[i]*points_vel[i].norm_sqr()+points_mass[i]*gravity.dot(domain[0]-points_pos[i])
  # 時間ステップの更新
  step[None] += 1
  time[None] = step[None]*dt

# 初期化する
initialize()

window_size = (640,640)
scale_to_pixel = window_size[0]/(domain[1]-domain[0]).x
gui = ti.GUI(os.path.basename(__file__),window_size)

while gui.running:
  # マウスの情報を取得する
  cursor_x,cursor_y = gui.get_cursor_pos()
  mouse_pos[None][0] = domain[0].x+(domain[1]-domain[0]).x*cursor_x
  mouse_pos[None][1] = domain[0].y+(domain[1]-domain[0]).y*cursor_y
  mouse_flag[None] = 0
  if gui.is_pressed(ti.GUI.LMB):
      mouse_flag[None] = 1
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
  if mouse_flag[None]:
    gui.circle((cursor_x,cursor_y),radius=mouse_radius*scale_to_pixel,color=0x800000)
  gui.circles(X,radius=R*scale_to_pixel,color=0x808080)
  gui.arrows(X,V,radius=2,color=0xFFFFFF) # 速度
  gui.arrows(X,A,radius=2,color=0xFF00FF) # 加速度
  gui.text(f'Step: {step[None]}, Time: {time[None]:.6f}, Energy: {energy[None]:.6f}',(0.0,1.0),font_size=20,color=0xFFFFFF)
  gui.show()
  