#include "nn_ops.h"
#include "platform.h"
#include <stdint.h>

int use_bram_accel = 1;
int use_conv_accel = 0;

static inline int16_t clamp_s16(int32_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

// Software sigmoid using piecewise linear approximation (no floating point)
// Input x is Q8.8, output is Q8.8 (range 0 to 256)
static inline int16_t sw_sigmoid(int16_t x_q8) {
    if (x_q8 <= -1024) return 0;   // x <= -4.0
    if (x_q8 >= 1024) return 256;  // x >= 4.0
    int32_t linear = (int32_t)x_q8 / 8 + 128;
    if (linear < 0) linear = 0;
    if (linear > 256) linear = 256;
    return (int16_t)linear;
}

static inline int16_t sw_relu(int16_t x) {
    return x > 0 ? x : 0;
}

void conv2d_q8_same(const int16_t *input, int ch_in, int h, int w,
                    const int16_t *kernel, const int16_t *bias, int ch_out,
                    int16_t *output) {
#ifdef RENODE_ACCEL
    if (use_conv_accel) {
        conv2d_accel(input, ch_in, h, w, kernel, bias, ch_out, output);
        return;
    }
#endif
    int kh = 3, kw = 3;
    int pad = 1;
    for (int oc = 0; oc < ch_out; oc++) {
        for (int oh = 0; oh < h; oh++) {
            for (int ow = 0; ow < w; ow++) {
                int32_t sum = 0;
                for (int ic = 0; ic < ch_in; ic++) {
                    for (int ky = 0; ky < kh; ky++) {
                        int in_y = oh + ky - pad;
                        if (in_y < 0 || in_y >= h) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            int in_x = ow + kx - pad;
                            if (in_x < 0 || in_x >= w) continue;
                            int in_idx = ic * h * w + in_y * w + in_x;
                            int k_idx = oc * ch_in * kh * kw + ic * kh * kw + ky * kw + kx;
                            sum += (int32_t)input[in_idx] * kernel[k_idx];
                        }
                    }
                }
                sum = (sum >> 8) + bias[oc];
                int out_idx = oc * h * w + oh * w + ow;
                output[out_idx] = clamp_s16(sum);
            }
        }
    }
}

void maxpool2d_q8(const int16_t *input, int ch, int h, int w, int16_t *output) {
    int out_h = h / 2;
    int out_w = w / 2;
    for (int c = 0; c < ch; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int16_t max = -32768;
                for (int ky = 0; ky < 2; ky++) {
                    for (int kx = 0; kx < 2; kx++) {
                        int in_idx = c * h * w + (oh * 2 + ky) * w + (ow * 2 + kx);
                        int16_t v = input[in_idx];
                        if (v > max) max = v;
                    }
                }
                int out_idx = c * out_h * out_w + oh * out_w + ow;
                output[out_idx] = max;
            }
        }
    }
}

void fc_q8(const int16_t *input, int in_dim, const int16_t *weight, int out_dim,
           const int16_t *bias, int16_t *output) {
    for (int o = 0; o < out_dim; o++) {
        int32_t sum = 0;
        for (int i = 0; i < in_dim; i++) {
            sum += (int32_t)input[i] * weight[o * in_dim + i];
        }
        sum = (sum >> 8) + bias[o];
        output[o] = clamp_s16(sum);
    }
}

void apply_activation(int16_t *data, int count, int func) {
    if (use_bram_accel && (func == ACCEL_RELU || func == ACCEL_SIGMOID)) {
        bram_lut_batch(data, (uint32_t)count, (uint32_t)func);
    } else {
        for (int i = 0; i < count; i++) {
            if (func == ACCEL_RELU) {
                data[i] = sw_relu(data[i]);
            } else if (func == ACCEL_SIGMOID) {
                data[i] = sw_sigmoid(data[i]);
            }
        }
    }
}
