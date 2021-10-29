# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from pattern import Circle,Checkers,Spectrum

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ch = Checkers(250, 25)
    ch.show()

    circle = Circle(500, 100, 120, 256)
    circle.show()

    spectrum = Spectrum(1024)
    spectrum.show()

