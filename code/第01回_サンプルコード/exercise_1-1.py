import taichi as ti
import numpy
import os
ti.init(arch=ti.cpu)

# 計算領域
domain = ti.Vector.field(2,ti.f32,shape=(2))
domain[0] = (-2.0,-1.0) # (x_min,y_min)
domain[1] = (2.0,1.0) # (x_max,y_max)

# 点の数
distance = 0.2
NX_points = int((domain[1]-domain[0]).x/distance+0.5)
NY_points = int((domain[1]-domain[0]).y/distance+0.5)
N_points = NX_points*NY_points
print(N_points)

# 位置ベクトル
points_pos = ti.Vector.field(2,ti.f32,shape=(N_points))

@ti.kernel
def initialize():
  for i in range(N_points):
    points_pos[i][0] = domain[0].x+distance*(i%NX_points+0.5)
    points_pos[i][1] = domain[0].y+distance*(i//NX_points+0.5)

initialize()

window_size = (640,320)
scale_to_pixel = window_size[0]/(domain[1]-domain[0]).x
gui = ti.GUI(os.path.basename(__file__),window_size)

X = points_pos.to_numpy()
X[:,0] = (X[:,0]-domain[0].x)/(domain[1]-domain[0]).x
X[:,1] = (X[:,1]-domain[0].y)/(domain[1]-domain[0]).y

while gui.running:
  gui.circles(X,radius=0.1*scale_to_pixel,color=0x0000FF)
  gui.show()