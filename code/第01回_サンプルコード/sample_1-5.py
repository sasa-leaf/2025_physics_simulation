import taichi as ti
import time

# GUIを作成
gui = ti.GUI('Window Title',(640,360))

start = time.time()

while gui.running:
  now = time.time()
  
  # テキストを描画
  gui.text(content=f'Hello! {now-start:.6f}',pos=[0.1,0.5],font_size=34,color=0xFFFFFF)
  
  # ウィンドウを表示
  gui.show()
  
  # 開始から5秒以上経過したらウィンドウを閉じる
  if now-start >= 5:
    gui.running = False
