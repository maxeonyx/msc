# coding: utf-8

import tensorflow as tf
from matplotlib import pyplot as plt

def reg(n, p):
    return tf.abs(n) ** p

l = tf.linspace(-2, 2, 100)
x, y = tf.meshgrid(l, l)

fig, ax = plt.subplots(4)
for i, p in enumerate([0.5, 0.9, 1., 2.]):
    z = reg(x, p) + reg(y, p)
    ax[i].contourf(x,y,z)
    ax[i].set_title(f"{p:.1f}-norm")
    ax[i].set_aspect('equal')
    
plt.show()
