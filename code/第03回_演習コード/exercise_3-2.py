import taichi as ti
import numpy
import math
import os
ti.init(arch=ti.cpu)

# 計算領域
domain = ti.Vector.field(2,ti.f32,shape=(2))
domain[0] = (-1.0,-1.0) # (x_min,y_min)
domain[1] = (1.0,1.0) # (x_max,y_max)

# 粒子サイズ
psize = 0.05 # 直径[m]

# 矩形形状の粒子配置を返す関数
def create_rectangle(center_x,center_y,width,height):
  array_pos = []
  Nx = int(width/psize+0.5)
  Ny = int(height/psize+0.5)
  for ix in range(Nx):
    for iy in range(Ny):
      x_i = center_x-width/2+psize*(ix+0.5)
      y_i = center_y-height/2+psize*(iy+0.5)
      array_pos.append([x_i,y_i])
  array_pos = numpy.array(array_pos,dtype=numpy.float32)
  return array_pos

# 重心を計算する関数
def calculate_center(pset):
  center = numpy.zeros(2,dtype=numpy.float32)
  for i in range(len(pset)):
    center += pset[i]
  center /= len(pset)
  return center

# 慣性モーメントを計算する関数
def calculate_inertia(pset,mass,center):
  inertia = 1.0
  # あとで実装する
  return inertia

# 剛体の数
N_rigids = 1

# 剛体のパラメータと初期条件
rigids_pset = [None]*N_rigids # 剛体形状を表す粒子集合
rigids_mass = ti.field(ti.f32,shape=(N_rigids)) # 質量[kg]
rigids_pos_ini = ti.Vector.field(2,ti.f32,shape=(N_rigids)) # 初期の重心位置
rigids_vel_ini = ti.Vector.field(2,ti.f32,shape=(N_rigids)) # 初期の重心速度[m/s]
rigids_angle_ini = ti.field(ti.f32,shape=(N_rigids)) # 初期の角度[rad]
rigids_omega_ini = ti.field(ti.f32,shape=(N_rigids)) # 初期の角速度[rad/s]
rigids_rcenter = ti.Vector.field(2,ti.f32,shape=(N_rigids)) # 回転中心
rigids_inertia = ti.field(ti.f32,shape=(N_rigids)) # 慣性モーメント

for k in range(N_rigids):
  rigids_pset[k] = create_rectangle(0.0,0.0,1.0,0.2)
  rigids_mass[k] = 2.0
  rigids_pos_ini[k] = calculate_center(rigids_pset[k])
  rigids_vel_ini[k] = (0.2,0.0)
  rigids_angle_ini[k] = math.radians(45.0)
  rigids_omega_ini[k] = math.radians(90.0)
  rigids_rcenter[k] = rigids_pos_ini[k]
  rigids_inertia[k] = calculate_inertia(rigids_pset[k],rigids_mass[k],rigids_rcenter[k])


# 剛体の変数
rigids_pos = ti.Vector.field(2,ti.f32,shape=(N_rigids)) # 現在の重心位置
rigids_vel = ti.Vector.field(2,ti.f32,shape=(N_rigids)) # 重心速度[m/s]
rigids_angle = ti.field(ti.f32,shape=(N_rigids)) # 角度[rad]
rigids_omega = ti.field(ti.f32,shape=(N_rigids)) # 角速度[rad/s]
rigids_rmatrix = ti.Matrix.field(2,2,ti.f32,shape=(N_rigids)) # 回転行列

# 初期粒子データ用の配列を作成する
array_rigid_id = [] # 粒子が属する剛体の番号
array_mass = [] # 粒子の質量
array_pos = [] # 粒子の位置ベクトル
for k,pset in enumerate(rigids_pset):
  for i in range(len(pset)):
    array_rigid_id.append(k)
    array_mass.append(rigids_mass[k]/len(pset))
    array_pos.append(pset[i])
array_rigid_id = numpy.array(array_rigid_id,dtype=numpy.int32)
array_mass = numpy.array(array_mass,dtype=numpy.float32)
array_pos = numpy.array(array_pos,dtype=numpy.float32)

# 粒子の数
N_particles = len(array_pos)

# 初期粒子データ
particles_rigid_id = ti.field(ti.i32,shape=(N_particles)) # 粒子が属する剛体の番号
particles_mass = ti.field(ti.f32,shape=(N_particles)) # 粒子の質量
particles_pos_ini = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 初期の位置ベクトル
particles_rigid_id.from_numpy(array_rigid_id)
particles_mass.from_numpy(array_mass)
particles_pos_ini.from_numpy(array_pos)

# 粒子の変数
particles_pos = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 位置ベクトル
particles_vel = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 速度ベクトル

# エネルギー
energy = ti.field(ti.f32,shape=())

# 時間
step = ti.field(ti.i32,shape=())
time = ti.field(ti.f32,shape=())
dt = ti.field(ti.f32,shape=())

# 初期化カーネル
@ti.kernel
def initialize():
  # 剛体の変数の初期化
  for k in range(N_rigids):
    rigids_angle[k] = rigids_angle_ini[k]
    rigids_omega[k] = rigids_omega_ini[k]
    rigids_rmatrix[k] = ti.math.rotation2d(rigids_angle[k])
    rigids_pos[k] = rigids_pos_ini[k]
    rigids_vel[k] = rigids_vel_ini[k]
    rigids_rcenter[k] = rigids_pos[k]
  # 粒子の変数の初期化
  for i in range(N_particles):
    k = particles_rigid_id[i]
    particles_pos[i] = rigids_pos[k]+rigids_rmatrix[k]@(particles_pos_ini[i]-rigids_pos_ini[k])
    particles_vel[i].x = rigids_vel[k].x - (particles_pos[i]-rigids_rcenter[k]).y*rigids_omega[k]
    particles_vel[i].y = rigids_vel[k].y + (particles_pos[i]-rigids_rcenter[k]).x*rigids_omega[k]
  # 時間の変数の初期化
  step[None] = 0
  time[None] = 0.0
  dt[None] = 0.01

# 更新カーネル
@ti.kernel
def update():
  # 重心位置と角度の更新
  for k in range(N_rigids):
    rigids_pos[k] += rigids_vel[k]*dt[None]
    rigids_angle[k] += rigids_omega[k]*dt[None]
    rigids_rmatrix[k] = ti.math.rotation2d(rigids_angle[k])
    rigids_rcenter[k] = rigids_pos[k]
  # 粒子位置と速度の更新
  for i in range(N_particles):
    k = particles_rigid_id[i]
    particles_pos[i] = rigids_pos[k]+rigids_rmatrix[k]@(particles_pos_ini[i]-rigids_pos_ini[k])
    particles_vel[i].x = rigids_vel[k].x - (particles_pos[i]-rigids_rcenter[k]).y*rigids_omega[k]
    particles_vel[i].y = rigids_vel[k].y + (particles_pos[i]-rigids_rcenter[k]).x*rigids_omega[k]
  # エネルギー
  energy[None] = 0.0
  for i in range(N_rigids):
    energy[None] += 0.5*rigids_mass[i]*particles_vel[i].norm_sqr()
  # 時間ステップの更新
  step[None] += 1
  time[None] += dt[None]

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
  # 粒子分布の描画
  X = particles_pos.to_numpy()
  X[:,0] = (X[:,0]-domain[0].x)/(domain[1]-domain[0]).x
  X[:,1] = (X[:,1]-domain[0].y)/(domain[1]-domain[0]).y
  gui.circles(X,radius=psize*0.5*scale_to_pixel,color=0x808080) # 剛体粒子
  # 回転中心の描画
  X = rigids_rcenter.to_numpy()
  X[:,0] = (X[:,0]-domain[0].x)/(domain[1]-domain[0]).x
  X[:,1] = (X[:,1]-domain[0].y)/(domain[1]-domain[0]).y
  gui.circles(X,radius=psize*0.35*scale_to_pixel,color=0xFFFF00) # 回転中心
  # 剛体重心の描画
  X = rigids_pos.to_numpy()
  X[:,0] = (X[:,0]-domain[0].x)/(domain[1]-domain[0]).x
  X[:,1] = (X[:,1]-domain[0].y)/(domain[1]-domain[0]).y
  gui.circles(X,radius=psize*0.25*scale_to_pixel,color=0xFF0000) # 重心
  gui.arrows(X,rigids_vel.to_numpy()*0.1,radius=2,color=0xFFFFFF) # 重心速度ベクトル
  # テキスト
  gui.text(f'Step: {step[None]}, Time: {time[None]:.6f}',(0.0,1.0),font_size=20,color=0xFFFFFF)
  gui.text(f'Particles: {N_particles}',(0.0,0.975),font_size=20,color=0xFFFFFF)
  gui.text(f'Energy: {energy[None]:.6f} J',(0.0,0.95),font_size=20,color=0xFFFFFF)
  gui.show()
  