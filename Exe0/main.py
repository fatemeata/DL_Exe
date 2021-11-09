import numpy as np
import matplotlib.pyplot as plt

from pattern import Circle, Checker, Spectrum
from generator import ImageGenerator


if __name__ == '__main__':

    ch = Checker(100, 10)
    ch.show()

    circle = Circle(250, 50, (120, 125))
    circle.show()

    spectrum = Spectrum(1024)
    spectrum.show()

    img = ImageGenerator("data\\exercise_data", "data\\labels.json", 10, [32, 32, 3],
                         rotation=True, mirroring=False, shuffle=True)
    img.show()