#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

#define stb__perlin_ease(a)   (((a*6-15)*a + 10) * a * a * a)

static float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

static float grad(int hash, float x, float y, float grads[4][2]) {
    int h = hash % 4;
    float gx = grads[h][0];
    float gy = grads[h][1];
    return gx * x + gy * y;
}

static int stb__perlin_fastfloor(float a)
{
    int ai = (int) a;
    return (a < ai) ? ai-1 : ai;
}

// __attribute__((visibility("default")))
void perlin_noise(int *perm, int w, int h, float scale, float grads[4][2], float *output) {
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < w; i++) {
        for (int j = 0; j < h; j++) {
            float x = (float)i / (float)w * scale;
            float y = (float)j / (float)h * scale;

            int xi = ((int)stb__perlin_fastfloor(x)) & 255;
            int yi = ((int)stb__perlin_fastfloor(y)) & 255 ;

            float xd = x - xi;
            float yd = y - yi;

            float u = stb__perlin_ease(xd);
            float v = stb__perlin_ease(yd);

            int aa = perm[perm[xi] + yi];
            int ab = perm[perm[xi] + (yi + 1)];
            int ba = perm[perm[(xi + 1)] + yi];
            int bb = perm[perm[(xi + 1)] + (yi + 1)];

            float n00 = grad(aa, xd, yd, grads);
            float n01 = grad(ab, xd, yd - 1, grads);
            float n10 = grad(ba, xd - 1, yd, grads);
            float n11 = grad(bb, xd - 1, yd - 1, grads);

            float x1 = lerp(n00, n10, u);
            float x2 = lerp(n01, n11, u);
            float noise = lerp(x1, x2, v);

            output[i * h + j] = noise;
        }
    }
}