import taichi as ti
import numpy

gui = ti.GUI('Window Title',(640,360))

red_circles = []
green_circles = []
blue_circles = []

while gui.running:
  mouse_x, mouse_y = gui.get_cursor_pos()
  gui.text(content=f'{mouse_x:.3f},{mouse_y:.3f}',pos=[0.0,1.0],font_size=34,color=0xFFFFFF)
  
  for e in gui.get_events(gui.PRESS):
    gui.text(content=f'key = {e.key}',pos=[0.0,0.9],font_size=34,color=0xFFFFFF)
    
    # クリックで円を追加
    if e.key == ti.GUI.LMB:
      red_circles.append([mouse_x,mouse_y])
    if e.key == ti.GUI.MMB:
      green_circles.append([mouse_x,mouse_y])
    if e.key == ti.GUI.RMB:
      blue_circles.append([mouse_x,mouse_y])
    
    # Escキーで初期化
    if e.key == ti.GUI.ESCAPE:
      red_circles = []
      green_circles = []
      blue_circles = []
    
  # 円を描画
  if len(red_circles) > 0:
    gui.circles(numpy.array(red_circles),radius=10,color=0xFF0000)
  if len(green_circles) > 0:
    gui.circles(numpy.array(green_circles),radius=10,color=0x00FF00)
  if len(blue_circles) > 0:
    gui.circles(numpy.array(blue_circles),radius=10,color=0x0000FF)
  
  gui.show()
