import taichi as ti
import numpy
import os
ti.init(arch=ti.cpu)

# 計算領域
domain = ti.Vector.field(2,ti.f32,shape=(2))
domain[0] = (-1.0,-1.0) # (x_min,y_min)
domain[1] = (1.0,1.0) # (x_max,y_max)

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

window_size = (640,640)
scale_to_pixel = window_size[0]/(domain[1]-domain[0]).x
gui = ti.GUI(os.path.basename(__file__),window_size)

initialize()

while gui.running:
  update()
  for e in gui.get_events(gui.PRESS):
    if e.key == ti.GUI.ESCAPE:
      initialize()
  X = points_pos.to_numpy()
  X[:,0] = (X[:,0]-domain[0].x)/(domain[1]-domain[0]).x
  X[:,1] = (X[:,1]-domain[0].y)/(domain[1]-domain[0]).y
  V = points_vel.to_numpy()*0.1
  gui.circles(X,radius=radius*scale_to_pixel,color=0x0000FF)
  gui.arrows(X,V,radius=2,color=0xFFFFFF)
  gui.text(f'Time: {time[None]:.6f}',(0.0,1.0),font_size=20,color=0xFFFFFF)
  gui.show()
