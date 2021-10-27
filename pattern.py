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
        merge = np.concatenate((blk, wht), axis=1)
        merge = np.concatenate((merge, np.flip(merge)), axis=0)
        print(merge)
        merge = np.tile(merge, (rep, rep))
        return merge

    def show(self):
        self.output = self.draw()
        plt.imshow(self.output, cmap='gray')
        plt.show()


# class Circle:


c = Checkers(32, 8)
c.show()
