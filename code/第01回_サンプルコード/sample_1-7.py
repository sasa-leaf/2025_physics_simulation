import taichi as ti

gui = ti.GUI('Window Title',(640,640))

while gui.running:
  # 1つの円を描画
  gui.circle((0.25,0.75),radius=10,color=0xFFFFFF)

  # 2つの線分を描画
  gui.line((0.0,0.5),(1.0,0.5),radius=2,color=0xFFFFFF)
  gui.line((0.5,0.0),(0.5,1.0),radius=2,color=0xFFFFFF)
  
  # 1つの矢印を描画
  gui.arrow((0.7,0.7),(0.1,0.1),radius=2,color=0xFFFFFF)
  
  # 1つの三角形を描画
  gui.triangle((0.3,0.2),(0.25,0.3),(0.2,0.2),color=0xFFFFFF)
  
  # 1つの矩形を描画
  gui.rect(topleft=(0.6,0.3),bottomright=(0.9,0.2),radius=2,color=0xFFFFFF)
  
  gui.show()
