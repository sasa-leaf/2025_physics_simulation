import taichi as ti
import numpy
import math
import os
ti.init(arch=ti.cpu) # CPUを使う場合
# ti.init(arch=ti.gpu) # GPUを使う場合

# 計算領域
domain = ti.Vector.field(2,ti.f32,shape=(2))
domain[0] = (-1.0,-1.0) # (x_min,y_min)
domain[1] = (1.0,1.0) # (x_max,y_max)

# 重力
gravity = ti.Vector([0.0,-3.0]) # 重力加速度[m/s^2]

# 流体の物性値
fluid_density = 1000.0 # 密度[kg/m^3]
fluid_viscosity = 0.001 # 粘性係数[Pa*s]
fluid_sound = 10.0 # （仮想的な）音速[m/s]

# 定数
psize = 0.0125 # 初期粒子間距離[m]
re = psize*2.5 # 影響半径
pnd0 = ti.field(ti.f32,shape=())
lambda0 = ti.field(ti.f32,shape=())
pnd0_gradP = ti.field(ti.f32,shape=())

# 衝突モデル
collision_dist = psize*0.9 # 衝突判定距離
collision_coef = 0.5 # 反発係数

# 粒子タイプ識別用定数
type_fluid = 0
type_wall = 1
type_ghost = 2
type_strong_wall = 3

# # 矩形形状の粒子配置を返す関数
# def create_rectangle(center_x,center_y,width,height):
#     array_pos = []
#     Nx = int(width/psize+0.5)
#     Ny = int(height/psize+0.5)
#     for ix in range(Nx):
#         for iy in range(Ny):
#             x_i = center_x-width/2+psize*(ix+0.5)
#             y_i = center_y-height/2+psize*(iy+0.5)
#             array_pos.append([x_i,y_i])
#     array_pos = numpy.array(array_pos,dtype=numpy.float32)
#     return array_pos

# くねった管（上部だけろうと型）の壁を作成する関数
# 川以外の部分をすべて埋める壁作成関数
def create_sine_pipe_wall(y_min, y_max, width_top, width_bottom, amplitude, frequency):
    array_pos = []
    array_type_local = [] 
    
    x_min_domain = -1.0
    x_max_domain = 1.0

    height = y_max - y_min
    if height <= 0: height = 1.0

    # 設定：上部何割を「ろうと」にするか
    funnel_ratio = 0.2 
    funnel_start_ratio = 1.0 - funnel_ratio

    num_y = int(height / (psize * 0.8))
    
    for i in range(num_y):
        y = y_max - i * (psize * 0.8) # 上から下へ
        
        ratio = (y - y_min) / height
        
        is_funnel_part = (ratio >= funnel_start_ratio)
        current_type = type_strong_wall if is_funnel_part else type_wall
        
        if not is_funnel_part:
            current_width = width_bottom
        else:
            funnel_progress = (ratio - funnel_start_ratio) / funnel_ratio
            current_width = width_bottom + (width_top - width_bottom) * funnel_progress

        center_x = amplitude * math.sin(frequency * y * math.pi)
        
        river_left_edge = center_x - current_width / 2.0
        river_right_edge = center_x + current_width / 2.0

        # --- 左側の埋め尽くし ---
        # こちらは x_min_domain (-1.0) から開始しているのでグリッドはずれない
        x_cursor = x_min_domain
        while x_cursor < river_left_edge:
            array_pos.append([x_cursor, y])
            array_type_local.append(current_type)
            x_cursor += psize

        # --- 右側の埋め尽くし ---
        # 【修正箇所】
        # 以前: x_cursor = river_right_edge (これがねじれの原因)
        # 修正: グリッド(-1.0基準)に乗るような位置を計算してスタートする
        
        # 「-1.0 から psize 刻みで進んで、river_right_edge を超えた最初の地点」を計算
        # x = -1.0 + k * psize >= river_right_edge
        k = math.ceil((river_right_edge - x_min_domain) / psize)
        x_cursor = x_min_domain + k * psize

        while x_cursor <= x_max_domain:
            array_pos.append([x_cursor, y])
            array_type_local.append(current_type)
            x_cursor += psize

    return numpy.array(array_pos, dtype=numpy.float32), numpy.array(array_type_local, dtype=numpy.int32)

# # 矩形タンク壁面の粒子配置を返す関数
# def create_rectangle_wall(center_x,center_y,width,height,layer=3):
#     array_pos = []
#     Nx = int(width/psize+0.5)
#     Ny = int(height/psize+0.5)
#     for ix in range(-layer,Nx+layer):
#         for iy in range(-layer,Ny+layer):
#             if 0 <= ix < Nx and 0 <= iy < Ny:
#                 continue
#             x_i = center_x-width/2+psize*(ix+0.5)
#             y_i = center_y-height/2+psize*(iy+0.5)
#             array_pos.append([x_i,y_i])
#     array_pos = numpy.array(array_pos,dtype=numpy.float32)
#     return array_pos

# 初期粒子データ用の配列を作成する
array_type = [] 
array_pos = []

# パイプ作成の呼び出し部分（★修正）
wall_positions, wall_types = create_sine_pipe_wall(
    domain[0].y, 
    domain[1].y, 
    width_top=0.5,    
    width_bottom=0.2, 
    amplitude=0.2,    
    frequency=1.0,
)

# 生成された壁とろうと粒子を追加
for i in range(len(wall_positions)):
    array_pos.append(wall_positions[i])
    array_type.append(wall_types[i])

N_space = 10000 # 流入粒子用の空きスロット数
for i in range(N_space):
    array_type.append(type_ghost)
    array_pos.append([0.0,0.0])

array_type = numpy.array(array_type,dtype=numpy.int32)
array_pos = numpy.array(array_pos,dtype=numpy.float32)
N_particles = len(array_pos) # 粒子数


# 初期粒子データ
particles_type_ini = ti.field(ti.i32,shape=(N_particles)) # 初期粒子タイプ
particles_pos_ini = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 初期の位置ベクトル
particles_type_ini.from_numpy(array_type)
particles_pos_ini.from_numpy(array_pos)


# 変数
particles_type = ti.field(ti.i32,shape=(N_particles)) # 粒子タイプ
particles_pos = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 位置ベクトル
particles_vel = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 速度ベクトル[m/s]
particles_force = ti.Vector.field(2,ti.f32,shape=(N_particles)) # 単位体積あたりの力[N/m^3]
particles_pnd = ti.field(ti.f32,shape=(N_particles)) # 粒子数密度
particles_pres = ti.field(ti.f32,shape=(N_particles)) # 圧力[Pa]
particles_color = ti.field(ti.i32,shape=(N_particles)) # 描画する色

# バケットデータ
Nx_buckets = int((domain[1]-domain[0]).x/re)+1
Ny_buckets = int((domain[1]-domain[0]).y/re)+1
N_buckets = Nx_buckets*Ny_buckets
cnt_max = 100 # バケット内の最大粒子数
table_cnt = ti.field(ti.i32,shape=(Nx_buckets,Ny_buckets))
table_data = ti.field(ti.i32,shape=(Nx_buckets,Ny_buckets,cnt_max))

# マウス変数
mouse_pos = ti.Vector.field(2,ti.f32,shape=())
mouse_state = ti.field(ti.i32,shape=()) # 0:押されていない, 1:左クリック

# 流体の流入口
array_pos = []
array_vel = []
for ix,iy in ti.ndrange((-10,11),(-10,11)):
    x_i = ix*psize
    y_i = iy*psize
    if x_i**2+y_i**2 < (psize*7)**2:
        array_pos.append([x_i,y_i])
        array_vel.append([0.0,-2.0])
array_pos = numpy.array(array_pos,dtype=numpy.float32)
array_vel = numpy.array(array_vel,dtype=numpy.float32)
N_injectors = len(array_pos)
injectors_pos = ti.Vector.field(2,ti.f32,shape=(N_injectors)) # 流入位置
injectors_vel = ti.Vector.field(2,ti.f32,shape=(N_injectors)) # 流入速度
injectors_pos.from_numpy(array_pos)
injectors_vel.from_numpy(array_vel)

# 安定条件
dt_max = 0.00125 # dtの上限値
courant_max = 0.1 # 最大クーラン数
diffusion_max = 0.1 # 最大拡散数

# 時間
step = ti.field(ti.i32,shape=())
time = ti.field(ti.f32,shape=())
dt = ti.field(ti.f32,shape=())
substeps = ti.field(ti.i32,shape=())
substeps_max = 50


# 重み関数
@ti.func
def weight(r) -> ti.f32:
    result = 0.0
    if r < re:
        result = re/r-1
    return result

# 圧力勾配用重み関数
@ti.func
def weight_gradP(r) -> ti.f32:
    result = 0.0
    if r < re:
        result = (1-r/re)**2
    return result

# バケットデータ更新関数
@ti.func
def bucket_update():
    for bx,by in ti.ndrange(Nx_buckets,Ny_buckets):
        table_cnt[bx,by] = 0
    for i in range(N_particles):
        if particles_type[i] == type_ghost:
            continue
        pos_i = particles_pos[i]
        bx = int((pos_i-domain[0]).x/re)
        by = int((pos_i-domain[0]).y/re)
        if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
            continue
        l = ti.atomic_add(table_cnt[bx,by],1)
        table_data[bx,by,l] = i

# 初期化カーネル
@ti.kernel
def initialize():
    # 定数係数の計算
    pnd0[None] = 0.0
    lambda0[None] = 0.0
    pnd0_gradP[None] = 0.0
    for jx,jy in ti.ndrange((-5,6),(-5,6)):
        if jx == 0 and jy == 0:
            continue
        pos_ij = ti.Vector([psize*jx,psize*jy])
        dist_ij = pos_ij.norm()
        w_ij = weight(dist_ij)
        pnd0[None] += w_ij
        lambda0[None] += w_ij*dist_ij**2
        pnd0_gradP[None] += weight_gradP(dist_ij)
    lambda0[None] /= pnd0[None]

    # 粒子の変数の初期化
    for i in range(N_particles):
        particles_type[i] = particles_type_ini[i]
        particles_pos[i] = particles_pos_ini[i]
        particles_vel[i] = (0.0,0.0)

    # 時間の変数の初期化
    step[None] = 0
    time[None] = 0.0
    dt[None] = dt_max
    substeps[None] = 1


# 事前計算カーネル
@ti.kernel
def preupdate():
    # 最大速度の計算
    vel_sqr_max = 0.0
    for i in range(N_particles):
        vel_sqr_i = particles_vel[i].norm_sqr()
        ti.atomic_max(vel_sqr_max,vel_sqr_i)
    vel_max = ti.math.sqrt(vel_sqr_max)
    acc_max = gravity.norm()
    dt[None] = min(dt_max,dt[None]*1.5)
    if vel_max > 0.0:
        dt[None] = min(dt[None],courant_max*psize/vel_max)
    if acc_max > 0.0:
        dt[None] = min(dt[None],ti.math.sqrt(courant_max*psize/acc_max))
    if fluid_viscosity > 0.0:
        dt[None] = min(dt[None],diffusion_max*psize**2*fluid_density/fluid_viscosity)
    substeps[None] = ti.math.ceil(dt_max/dt[None],dtype=ti.i32)
    if substeps[None] > substeps_max:
        substeps[None] = substeps_max
    else:
        dt[None] = dt_max/substeps[None]

# 更新カーネル
@ti.kernel
def update():
    # バケットの更新
    bucket_update()

    # 力の計算（1回目）
    for i in range(N_particles):
        particles_force[i].fill(0.0)
        if particles_type[i] == type_wall or particles_type[i] == type_ghost:
            continue
        # 重力
        if particles_type[i] == type_fluid:
            particles_force[i] += fluid_density * gravity

        # 近傍粒子探索ループ
        bx0 = int((particles_pos[i]-domain[0]).x/re)
        by0 = int((particles_pos[i]-domain[0]).y/re)
        for bx,by in ti.ndrange((bx0-1,bx0+2),(by0-1,by0+2)):
            if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                continue
            for l in range(table_cnt[bx,by]):
                j = table_data[bx,by,l]
                if j == i:
                    continue
                pos_ij = particles_pos[j]-particles_pos[i]
                dist_ij = pos_ij.norm()
                # 粘性項
                if particles_type[i] == type_fluid or particles_type[j] == type_fluid:
                    if dist_ij < re:
                        particles_force[i] += fluid_viscosity*4.0/(pnd0[None]*lambda0[None])*(particles_vel[j]-particles_vel[i])*weight(dist_ij)

    # 粒子の仮速度と仮位置
    for i in range(N_particles):
        if particles_type[i] == type_fluid:
            particles_vel[i] += (particles_force[i]/fluid_density)*dt[None]
            particles_pos[i] += particles_vel[i]*dt[None]

    # 流入
    if mouse_state[None] == 1:
        for i in range(N_injectors):
            # 流入口に近接する粒子を探す
            j_min = -1
            dist_sqr_min = re**2
            pos_i = mouse_pos[None]+injectors_pos[i]
            bx0 = int((pos_i-domain[0]).x/re)
            by0 = int((pos_i-domain[0]).y/re)
            for bx,by in ti.ndrange((bx0-1,bx0+2),(by0-1,by0+2)):
                if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                    continue
                for l in range(table_cnt[bx,by]):
                    j = table_data[bx,by,l]
                    pos_ij = particles_pos[j]-pos_i
                    dist_sqr_ij = ti.math.dot(pos_ij,pos_ij)
                    if dist_sqr_min > dist_sqr_ij:
                        dist_sqr_min = dist_sqr_ij
                        j_min = j
            # 流入口に重なる粒子がある場合は流入せず、その粒子の速度を流入速度に書き換える
            if dist_sqr_min < (psize*0.99)**2:
                if particles_type[j_min] == type_fluid:
                    particles_vel[j_min] = injectors_vel[i]
                continue
            # 空きスロットを探す
            j_ghost = -1
            for j in range(N_particles):
                if particles_type[j] == type_ghost:
                    type_j = ti.atomic_min(particles_type[j],type_fluid)
                    if type_j == type_ghost:
                        j_ghost = j
                        break
            # 空きスロットがあれば粒子を発生させる
            if j_ghost != -1:
                particles_type[j_ghost] = type_fluid
                particles_vel[j_ghost] = injectors_vel[i]
                particles_pos[j_ghost] = pos_i

    # バケットの更新
    bucket_update()

    # 粒子数密度と圧力
    for i in range(N_particles):
        particles_pnd[i] = 0.0
        particles_pres[i] = 0.0
        if particles_type[i] == type_ghost:
            continue
        # 近傍粒子探索ループ
        bx0 = int((particles_pos[i]-domain[0]).x/re)
        by0 = int((particles_pos[i]-domain[0]).y/re)
        for bx,by in ti.ndrange((bx0-1,bx0+2),(by0-1,by0+2)):
            if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                continue
            for l in range(table_cnt[bx,by]):
                j = table_data[bx,by,l]
                if j == i:
                    continue
                pos_ij = particles_pos[j]-particles_pos[i]
                dist_ij = pos_ij.norm()
                # 粒子数密度
                if dist_ij < re:
                    particles_pnd[i] += weight(dist_ij)
        # 圧力
        if particles_pnd[i] > pnd0[None]:
            particles_pres[i] = fluid_density*fluid_sound**2*(particles_pnd[i]-pnd0[None])/pnd0[None]
        else:
            particles_pres[i] = 0.0

    # 力の計算（2回目）
    for i in range(N_particles):
        particles_force[i].fill(0.0)
        if particles_type[i] == type_wall or particles_type[i] == type_ghost:
            continue
        # 近傍粒子探索ループ
        bx0 = int((particles_pos[i]-domain[0]).x/re)
        by0 = int((particles_pos[i]-domain[0]).y/re)
        for bx,by in ti.ndrange((bx0-1,bx0+2),(by0-1,by0+2)):
            if bx < 0 or bx >= Nx_buckets or by < 0 or by >= Ny_buckets:
                continue
            for l in range(table_cnt[bx,by]):
                j = table_data[bx,by,l]
                if j == i:
                    continue
                pos_ij = particles_pos[j]-particles_pos[i]
                dist_ij = pos_ij.norm()
                # 圧力項
                if particles_type[i] == type_fluid or particles_type[j] == type_fluid:
                    if dist_ij < re:
                        particles_force[i] -= 2.0/pnd0_gradP[None]*(particles_pres[i]+particles_pres[j])*pos_ij/dist_ij**2*weight_gradP(dist_ij)
                # 衝突モデル
                if particles_type[i] == type_fluid or particles_type[j] == type_fluid:
                    if dist_ij < collision_dist:
                        vel_ij = particles_vel[j]-particles_vel[i]
                        normal_ij = pos_ij/dist_ij
                        tmp = normal_ij.dot(vel_ij)
                        if tmp < 0.0:
                            if (particles_type[i],particles_type[j]) == (type_fluid,type_fluid):
                                m_ij = fluid_density*0.5
                                particles_force[i] += normal_ij*(1.0+collision_coef)*m_ij*tmp/dt_max
                            elif (particles_type[i],particles_type[j]) == (type_fluid,type_wall):
                                m_ij = fluid_density
                                particles_force[i] += normal_ij*(1.0+collision_coef)*m_ij*tmp/dt_max
    
    # 粒子の速度と位置の修正
    for i in range(N_particles):
        if particles_type[i] == type_fluid:
            particles_vel[i] += (particles_force[i]/fluid_density)*dt[None]
            particles_pos[i] += (particles_force[i]/fluid_density)*dt[None]**2

    # 領域外粒子の処理
    for i in range(N_particles):
        if particles_type[i] != type_fluid:
            continue
        if particles_pos[i].x < domain[0].x or particles_pos[i].x >= domain[1].x or particles_pos[i].y < domain[0].y or particles_pos[i].y >= domain[1].y:
            particles_type[i] = type_ghost
            particles_pos[i] = (0.0,0.0)
            particles_vel[i] = (0.0,0.0)

    # 時間ステップを進める
    step[None] += 1
    time[None] += dt[None]


# 色計算カーネル
@ti.kernel
def update_colors():
    for i in range(N_particles):
        if particles_type[i] == type_fluid:
            a = ti.math.clamp(particles_vel[i].norm(),0.0,1.0)
            r = a
            b = 1.0-a
            g = 0.0
            particles_color[i] = 0x010000*ti.i32(r*255)+0x000100*ti.i32(g*255)+0x000001*ti.i32(b*255)
        elif particles_type[i] == type_wall:
            particles_color[i] = 0x808080
        else:
            particles_color[i] = 0xFFFFFF

# 初期化する
initialize()
print('pnd0 =',pnd0[None])
print('lambda0 =',lambda0[None])

# GUI作成
window_size = (640,640)
scale_to_pixel = window_size[0]/(domain[1]-domain[0]).x
gui = ti.GUI(os.path.basename(__file__),window_size)


# ウィジェット
slider_forwards = gui.slider('fast-forward',1,20)
slider_forwards.value = 2


while gui.running:
    # マウスの情報を取得する
    cursor_x,cursor_y = gui.get_cursor_pos()
    mouse_pos[None][0] = domain[0].x+(domain[1]-domain[0]).x*cursor_x
    mouse_pos[None][1] = domain[0].y+(domain[1]-domain[0]).y*cursor_y
    mouse_state[None] = 0
    if gui.is_pressed(ti.GUI.LMB):
        mouse_state[None] = 1

    # 時間を進める
    forwards = (int)(slider_forwards.value)
    slider_forwards.value = forwards
    for frame in range(forwards):
        preupdate()
        for substep in range(substeps[None]):
            update()
    # キーボード入力の受け取り
    for e in gui.get_events(gui.PRESS):
        if e.key == ti.GUI.ESCAPE:
            initialize()

    # 現在の状態を描画する
    update_colors()
    if mouse_state[None] == 1:
        J = injectors_pos.to_numpy()
        J[:,0] = (J[:,0]+mouse_pos[None].x-domain[0].x)/(domain[1]-domain[0]).x
        J[:,1] = (J[:,1]+mouse_pos[None].y-domain[0].y)/(domain[1]-domain[0]).y
        gui.circles(J,radius=psize*0.5*scale_to_pixel,color=0x00FF00)
    X = particles_pos.to_numpy()
    X[:,0] = (X[:,0]-domain[0].x)/(domain[1]-domain[0]).x
    X[:,1] = (X[:,1]-domain[0].y)/(domain[1]-domain[0]).y

    T = particles_type.to_numpy()
    C = particles_color.to_numpy()
    gui.circles(X[(T!=type_ghost),:],radius=psize*0.5*scale_to_pixel,color=C[(T!=type_ghost)])
    
    gui.text(f'Step: {step[None]}, Time: {time[None]:.6f}, substeps = {substeps[None]}',(0.0,1.0),font_size=20,color=0xFFFFFF)
    gui.text(f'Particles: {numpy.count_nonzero(T==type_fluid)} / {numpy.count_nonzero(T==type_wall)} / {numpy.count_nonzero(T==type_ghost)}',(0.0,0.975),font_size=20,color=0xFFFFFF)
    gui.show()
