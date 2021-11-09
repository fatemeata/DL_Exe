import numpy as np
import matplotlib.pyplot as plt


class Checker:

    def __init__(self, res, tsize):
        self.res = res
        self.t_size = tsize
        self.output = np.zeros((res, res))

    def draw(self):
        if self.res % (2 * self.t_size) != 0:
            print("Cannot draw the checkerboard!")

        else:
            blk = np.zeros((self.t_size, self.t_size))
            wht = np.ones((self.t_size, self.t_size))
            merge = np.hstack((blk, wht))
            merge = np.vstack((merge, np.flip(merge)))
            rep = int((self.res / self.t_size) / 2)
            self.output = np.tile(merge, (rep, rep))

        return self.output.copy()

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Circle:

    def __init__(self, res, r, xy):
        self.res = res
        self.rad = r
        self.x_0 = xy[0]
        self.y_0 = xy[1]
        self.output = np.zeros((res, res))

    def draw(self):
        x_r = np.arange(0, self.res, 1)  # x-axis
        y_r = np.arange(0, self.res, 1)  # y-axis

        # create the cartesian coordinate system
        xx, yy = np.meshgrid(x_r, y_r, indexing='xy')

        # create the circle with the coordinate (x_0, y_0)
        self.output = (((xx - self.x_0) ** 2) + ((yy - self.y_0) ** 2) <= self.rad ** 2).astype(int)
        return self.output.copy()

    def show(self):
        plt.imshow(self.draw(), cmap='gray')
        plt.show()


class Spectrum:

    def __init__(self, res):
        self.res = res
        self.output = np.zeros((res, res, 3))

    def draw(self):

        img = np.zeros((self.res, self.res, 3))

        # Red channel/ required matrix:
        # [0.............1]
        # [0.............1]
        # [...............]
        # [0.............1]

        r_vec = (np.linspace(1, 1, self.res)).reshape((self.res, 1))
        r_vec_t = (np.linspace(0, 1, self.res)).reshape((1, self.res))
        red_ch = np.matmul(r_vec, r_vec_t)

        img[:, :, 0] = red_ch

        # Green channel/ required matrix:
        # [0.. ...........0]
        # [0. .............]
        # [...............]
        # [1 1............1]

        g_vec = (np.linspace(0, 1, self.res)).reshape((self.res, 1))
        g_vec_t = ((np.linspace(1, 1, self.res)).reshape((1, self.res)))
        green_ch = np.matmul(g_vec, g_vec_t)

        img[:, :, 1] = green_ch

        # Blue channel/ required matrix:
        # [1.............0]
        # [1..............]
        # [...............]
        # [1.............0]

        b_vec = (np.linspace(1, 1, self.res)).reshape((self.res, 1))
        b_vec_t = (np.linspace(1, 0, self.res)).reshape((1, self.res))
        blue_ch = np.matmul(b_vec, b_vec_t)

        img[:, :, 2] = blue_ch

        self.output = img

        return self.output.copy()

    def show(self):
        plt.imshow(self.draw())
        plt.show()

