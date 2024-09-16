import torch

def get_grid(size, device, scale):
    w,h = size
    x = torch.linspace(0, scale, w).to(device)
    y = torch.linspace(0, scale, h).to(device)
    x, y = torch.meshgrid(x, y, indexing="ij")
    
    return x,y

def torch_perlin_noise(perm, x, y, grads):

    def fade(f):
        return 6 * f**5 - 15 * f**4 + 10 * f**3

    def lerp(a, b, t):
        return a + t * (b - a)

    def grad(c, x, y, grads):
        gradient_co = grads[c % 4]
        return gradient_co[:, :, 0] * x + gradient_co[:, :, 1] * y

    xi, yi = torch.floor(x).to(torch.int64) % 256, torch.floor(y).to(torch.int64) % 256

    xd, yd = x - xi, y - yi

    u, v = fade(xd), fade(yd)
    n00 = grad(perm[perm[xi] + yi], xd, yd, grads)
    n01 = grad(perm[perm[xi] + ((yi + 1))], xd, yd - 1, grads)
    n11 = grad(perm[perm[(xi + 1)] + (yi + 1)], xd - 1, yd - 1, grads)
    n10 = grad(perm[perm[(xi + 1)] + yi], xd - 1, yd, grads)

    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    noise = lerp(x1, x2, v)

    return noise.cpu()