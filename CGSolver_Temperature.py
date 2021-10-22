#温度泊松方程的求解器
import taichi as ti
@ti.data_oriented
class CGSolver_Temperature:
    def __init__(self, m, n,t_n ):
        self.m = m
        self.n = n
        self.t_n = t_n

        # 右侧的线性系统：
        self.b = ti.field(dtype=ti.f32, shape=(self.m, self.n))

        # 左侧的线性系统
        self.Adiag = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.Ax = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.Ay = ti.field(dtype=ti.f32, shape=(self.m, self.n))

        # cg需要的参数
        self.T = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        # r:残差
        self.r = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        #搜索方向
        self.d = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.Ad = ti.field(dtype=ti.f32, shape=(self.m, self.n))
        self.sum = ti.field(dtype=ti.f32, shape=())
        self.alpha = ti.field(dtype=ti.f32, shape=())
        self.beta = ti.field(dtype=ti.f32, shape=())

    @ti.kernel
    def system_init_kernel(self, scale: ti.f32):
        # 右边线性系统
        for i, j in ti.ndrange(self.m, self.n):
            self.b[i, j]=self.t_n[i,j]
            # 左侧线性系统：
        for i, j in ti.ndrange(self.m, self.n):
            # 对称
            count=0
            if  i-1 >= 0:
                count += 1
            if i+1 < self.m:
                count += 1
                self.Ax[i, j] = -scale
            if j-1 >= 0:
                count += 1
            if j+1 < self.n:
                count += 1
                self.Ay[i, j] = -scale

            self.Adiag[i, j]=1+count*scale

    def system_init(self, scale):
        self.b.fill(0)
        self.Adiag.fill(0.0)
        self.Ax.fill(0.0)
        self.Ay.fill(0.0)
        self.system_init_kernel(scale)

    def solve(self, max_iters):

        tol = 1e-12
        self.T.fill(0.0)
        self.Ad.fill(0.0)
        self.d.fill(0.0)

        # 该系统从原点出发
        self.r.copy_from(self.b)
        self.reduce(self.r, self.r)
        init_rTr = self.sum[None]

        if init_rTr < tol:
            print("init_rTr：",init_rTr)

        else:
            self.d.copy_from(self.r)
            old_rTr = init_rTr
            for i in range(max_iters):
                # Ad=A*d
                self.compute_Ad()
                # dTq
                self.reduce(self.d, self.Ad)
                dAd = self.sum[None]
                if dAd == 0:
                    break
                self.alpha[None] = old_rTr / dAd
                # T = T + alpha * d
                self.update_T()
                # r = r - alpha * As

                if i%10==0:
                    self.update_r1()
                else:
                    self.update_r2()

                # 检查收敛性
                self.reduce(self.r, self.r)
                rTr = self.sum[None]

                if rTr < init_rTr * tol:
                    break

                new_rTr = rTr
                self.beta[None] = new_rTr / old_rTr
                # d = r + beta * d
                self.update_d()
                old_rTr = new_rTr
                iteration =i

            # print("Converged to {} in {} iterations".format(rTr, iteration))

    @ti.kernel
    def reduce(self, p: ti.template(), q: ti.template()):
        self.sum[None] = 0.0
        for i, j in ti.ndrange(self.m, self.n):
                self.sum[None] += p[i, j] * q[i, j]

    @ti.kernel
    def compute_Ad(self):
        for i, j in ti.ndrange(self.m, self.n):
            if i - 1 >= 0 and j - 1 >= 0 and i + 1 <= self.m - 1 and j + 1 <= self.n - 1:
                self.Ad[i, j] = self.Adiag[i, j] * self.d[i, j] + self.Ax[
                    i - 1, j] * self.d[i - 1, j] + self.Ax[i, j] * self.d[
                                    i + 1, j] + self.Ay[i, j - 1] * self.d[
                                    i, j - 1] + self.Ay[i, j] * self.d[i, j + 1]

    @ti.kernel
    def update_T(self):
        for i, j in ti.ndrange(self.m, self.n):
            self.T[i, j] = self.T[i, j] + self.alpha[None] * self.d[i, j]

    @ti.kernel
    def update_r1(self):
        for i, j in ti.ndrange(self.m, self.n):
            if i - 1 >= 0 and j - 1 >= 0 and i + 1 <= self.m - 1 and j + 1 <= self.n - 1:
                self.r[i, j] = self.b[i, j] - (self.Adiag[i, j] * self.T[i, j] + self.Ax[
                    i - 1, j] * self.T[i - 1, j] + self.Ax[i, j] * self.T[
                                    i + 1, j] + self.Ay[i, j - 1] * self.T[
                                    i, j - 1] + self.Ay[i, j] * self.T[i, j + 1])

    @ti.kernel
    def update_r2(self):
        for i, j in ti.ndrange(self.m, self.n):
            self.r[i, j] = self.r[i, j] - self.alpha[None] * self.Ad[i, j]

    @ti.kernel
    def update_d(self):
        for i, j in ti.ndrange(self.m, self.n):
            self.d[i, j] = self.r[i, j] + self.beta[None] * self.d[i, j]

