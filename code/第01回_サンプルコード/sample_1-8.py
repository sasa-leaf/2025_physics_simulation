import taichi as ti

gui = ti.GUI('Window Title',(640,640))

# 3つのスライダーの追加
slider_pos_x = gui.slider('X',0.0,1.0)
slider_pos_y = gui.slider('Y',0.0,1.0)
slider_radius = gui.slider('Radius',1,50)

# スライダーの初期値
slider_pos_x.value = 0.5
slider_pos_y.value = 0.5
slider_radius.value = 10

while gui.running:
  gui.circle((slider_pos_x.value,slider_pos_y.value),radius=slider_radius.value)
  gui.show()