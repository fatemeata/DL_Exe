import numpy as np
import matplotlib.pyplot as plt


class Checkers:
    reso = 1
    t_size = 1
    output = np.zeros((reso, reso))

    def __init__(self, res, tsize):
        self.reso = res
        self.t_size = tsize

    def draw(self):
        rep = int((self.reso / self.t_size) / 2)
        blk = np.zeros((self.t_size, self.t_size))
        wht = np.ones((self.t_size, self.t_size))
        merge = np.hstack((blk, wht))
        merge = np.vstack((merge, np.flip(merge)))
        self.output = np.tile(merge, (rep, rep))
        merge = self.output
        return merge

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Circle:


    def __init__(self, res, r, x, y):
        self.res = res
        self.rad = r
        self.x_0 = x
        self.y_0 = y
        self.output = np.zeros((res, res))

    def draw(self):
        background = np.zeros((self.res, self.res))

        x_r = np.arange(0, self.res, 1)
        y_r = np.arange(0, self.res, 1)

        xx, yy = np.meshgrid(x_r, y_r, indexing='xy')

        self.output = ((xx - self.x_0) ** 2) + ((yy - self.y_0) ** 2) <= self.rad ** 2
        z = self.output
        return z

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Spectrum:
    res = 0

    def __init__(self, res):
        self.res = res
        self.output = np.zeros((res, res, 3))

    def draw(self):
        img = np.zeros((self.res, self.res, 3))

        ## Blue channel
        ## Blue channel/ required matrix:
        ## [1.............0]
        ## [1..............]
        ## [...............]
        ## [1.............0]

        b_vec = (np.linspace(1, 1, self.res)).reshape((self.res, 1))
        b_vec_t = (np.linspace(1, 0, self.res)).reshape((1, self.res))
        blue_ch = np.matmul(b_vec, b_vec_t)

        img[:, :, 2] = blue_ch

        ## Red channel/ required matrix:
        ## [0.............1]
        ## [0.............1]
        ## [0.............1]

        r_vec = (np.linspace(1, 1, self.res)).reshape((self.res, 1))
        r_vec_t = (np.linspace(0, 1, self.res)).reshape((1, self.res))
        red_ch = np.matmul(r_vec, r_vec_t)

        img[:, :, 0] = red_ch

        ## Green channel/ required matrix:
        ## [0.. ...........0]
        ## [0. .............]
        ## [1 1............1]

        g_vec = (np.linspace(0, 1, self.res)).reshape((self.res, 1))
        g_vec_t = ((np.linspace(1, 1, self.res)).reshape((1, self.res)))
        green_ch = np.matmul(g_vec, g_vec_t)

        img[:, :, 1] = green_ch

        self.output = img

        return self.output.copy()


    def show(self):

        plt.imshow(self.draw())
        plt.show()



