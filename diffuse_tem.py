import taichi as ti
from src.CGSolver_Temperature import CGSolver_Temperature

ti.init(arch=ti.gpu)


paused = True
save_images = False
n = 64
scatter = 8
res = n * scatter
h = 1e-3
substep = 1
dx = 1

# heat-source related
t_max = 300
t_min = 0
heat_center = (n // 2, n // 2)
heat_radius = 5.1
k = 2000.0

# visualization
pixels = ti.Vector.field(3, ti.f32, shape=(res, res))

t_n = ti.field(ti.f32, shape = (n,n))
t_np1 = ti.field(ti.f32, shape = (n,n))

ind = lambda i, j: i * n + j

poisson_solver=CGSolver_Temperature(n,n,t_n)

@ti.kernel
def init():
    for i,j in t_n:
        if (float(i)-heat_center[0])**2 + (float(j)-heat_center[1])**2 <= heat_radius**2:
            t_n[i, j] = t_max # source
            t_np1[i, j] = t_max # source
        else:
            t_n[i, j] = t_min
            t_np1[i, j] = t_min


@ti.kernel
def update_source():
    for i,j in t_n:
        if (float(i)-heat_center[0])**2 + (float(j)-heat_center[1])**2 <= heat_radius**2:
            t_np1[i, j] = t_max


def diffuse(dt: ti.f32):
    scale=k*dt/dx**2
    poisson_solver.system_init(scale)
    poisson_solver.solve(500)
    t_np1.copy_from(poisson_solver.T)

def update_source_and_commit():
    update_source()
    t_n.copy_from(t_np1)


@ti.func
def get_color(v, vmin, vmax):
    c = ti.Vector([1.0, 1.0, 1.0]) # white
    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    dv = vmax - vmin
    #把颜色分为四个阶段
    if v < (vmin + 0.25 * dv):
        c[0] = 0
        c[1] = 4 * (v-vmin) / dv
    elif v < (vmin + 0.5 * dv):
        c[0] = 0
        c[2] = 1 + 4 * (vmin + 0.25*dv -v) / dv
    elif v < (vmin + 0.75*dv):
        c[0] = 4 * (v - vmin -0.5 * dv) / dv
        c[2] = 0
    else:
        c[1] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
        c[2] = 0

    return c

@ti.kernel
def temperature_to_color(t: ti.template(), color: ti.template(), tmin: ti.f32, tmax: ti.f32):
    for i,j in t:
        for k,l in ti.ndrange(scatter, scatter):
            color[i*scatter+k,j*scatter+l] = get_color(t[i,j], tmin, tmax)


# GUI
my_gui = ti.GUI("Diffuse", (res, res))

init()
i = 0
while my_gui.running:

    for e in my_gui.get_events(ti.GUI.PRESS):
        if e.key == ti.GUI.ESCAPE:
            exit()
        elif e.key == ti.GUI.SPACE:
            paused = not paused
        elif e.key == 'i':
            save_images = not save_images
            # print(f"Exporting images to images\output_{i:05}.png")
        elif e.key == 'r':
            init()
            i = 0

    if not paused:
        for sub in range(substep):
            diffuse(h / substep)
            update_source_and_commit()

    temperature_to_color(t_np1, pixels, t_min, t_max)
    my_gui.set_image(pixels)
    if save_images and not paused:
        # my_gui.show(f"images\output_{i:05}.png")
        my_gui.show()
        i += 1
    else:
        my_gui.show()