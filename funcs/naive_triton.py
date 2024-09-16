import triton
import triton.language as tl
import torch

@triton.jit
def perlin_noise_triton_kernel(x_ptr, y_ptr, perm_ptr, grads_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    xi = tl.floor(x).to(tl.int8) & 255
    yi = tl.floor(y).to(tl.int8) & 255

    xd = x - xi
    yd = y - yi

    f3_x = xd * xd * xd
    f4_x = f3_x * xd
    f5_x = f4_x * xd
    u = 6 * f5_x - 15 * f4_x + 10 * f3_x

    f3_y = yd * yd * yd
    f4_y = f3_y * yd
    f5_y = f4_y * yd
    v = 6 * f5_y - 15 * f4_y + 10 * f3_y

    perm_x0 = tl.load(perm_ptr + xi , mask=mask)
    perm_x1 = tl.load(perm_ptr + (xi + 1), mask=mask)

    perm_00 = tl.load(perm_ptr + perm_x0 + yi , mask=mask)
    perm_01 = tl.load(perm_ptr + perm_x0 + yi + 1 , mask=mask)
    perm_10 = tl.load(perm_ptr + perm_x1 + yi, mask=mask)
    perm_11 = tl.load(perm_ptr + perm_x1 + yi + 1, mask=mask)
    
    grad_00_x = tl.load(grads_ptr + (perm_00 % 4)*2, mask=mask)
    grad_00_y = tl.load(grads_ptr + (perm_00 % 4)*2 + 1, mask=mask)

    grad_01_x = tl.load(grads_ptr + (perm_01 % 4)*2, mask=mask)
    grad_01_y = tl.load(grads_ptr + (perm_01 % 4)*2 + 1, mask=mask)

    grad_10_x = tl.load(grads_ptr + (perm_10 % 4)*2, mask=mask)
    grad_10_y = tl.load(grads_ptr + (perm_10 % 4)*2 + 1, mask=mask)

    grad_11_x = tl.load(grads_ptr + (perm_11 % 4)*2, mask=mask)
    grad_11_y = tl.load(grads_ptr + (perm_11 % 4)*2 + 1, mask=mask)

    dot_00 = grad_00_x * xd + grad_00_y * yd
    dot_01 = grad_01_x * xd + grad_01_y * (yd - 1)
    dot_10 = grad_10_x * (xd - 1) + grad_10_y * yd
    dot_11 = grad_11_x * (xd - 1) + grad_11_y * (yd - 1)

    lerp_x1 = dot_00 + u * (dot_10 - dot_00)
    lerp_x2 = dot_01 + u * (dot_11 - dot_01)
    noise = lerp_x1 + v * (lerp_x2 - lerp_x1)

    tl.store(out_ptr + offsets, noise, mask=mask)

def perlin_noise_triton(x: torch.Tensor, y: torch.Tensor, perm: torch.Tensor, grads: torch.Tensor, out: torch.Tensor, BLOCK, grid, n_elements):
    perlin_noise_triton_kernel[grid](x, y, perm, grads, out, n_elements, BLOCK_SIZE=BLOCK)
    return out.cpu()