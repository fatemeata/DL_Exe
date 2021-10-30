import numpy as np
import matplotlib.pyplot as plt

from pattern import Circle, Checkers, Spectrum


if __name__ == '__main__':

    ch = Checkers(250, 25)
    ch.show()

    circle = Circle(500, 100, 120, 256)
    circle.show()

    spectrum = Spectrum(1024)
    spectrum.show()

