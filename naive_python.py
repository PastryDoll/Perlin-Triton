import math
from numba import jit

@jit(nopython=True)
def perlin_ease(a):
    return ((a * 6 - 15) * a + 10) * a * a * a
@jit(nopython=True)
def lerp(a, b, t):
    return a + t * (b - a)
@jit(nopython=True)
def grad(hash, x, y, grads):
    h = hash % 4
    gx, gy = grads[h]
    return gx * x + gy * y
@jit(nopython=True)
def perlin_noise(perm, w, h, scale, grads):
    output = [0.0] * (w * h)
    
    for i in range(w):
        for j in range(h):
            x = float(i) / float(w) * scale
            y = float(j) / float(h) * scale

            xi = math.floor(x) % 256
            yi = math.floor(y) % 256

            xd = x - math.floor(x)
            yd = y - math.floor(y)

            u = perlin_ease(xd)
            v = perlin_ease(yd)

            aa = perm[perm[xi] + yi]
            ab = perm[perm[xi] + (yi + 1)]
            ba = perm[perm[(xi + 1)] + yi]
            bb = perm[perm[(xi + 1)] + (yi + 1)]

            n00 = grad(aa, xd, yd, grads)
            n01 = grad(ab, xd, yd - 1, grads)
            n10 = grad(ba, xd - 1, yd, grads)
            n11 = grad(bb, xd - 1, yd - 1, grads)

            x1 = lerp(n00, n10, u)
            x2 = lerp(n01, n11, u)
            noise = lerp(x1, x2, v)

            output[i * h + j] = noise
    
    return output