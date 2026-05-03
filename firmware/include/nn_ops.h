#ifndef NN_OPS_H
#define NN_OPS_H

#include <stdint.h>

extern int use_bram_accel;
extern int use_conv_accel;

void conv2d_q8_same(const int16_t *input, int ch_in, int h, int w,
                    const int16_t *kernel, const int16_t *bias, int ch_out,
                    int16_t *output);

void maxpool2d_q8(const int16_t *input, int ch, int h, int w, int16_t *output);

void fc_q8(const int16_t *input, int in_dim, const int16_t *weight, int out_dim,
           const int16_t *bias, int16_t *output);

void apply_activation(int16_t *data, int count, int func);

#endif
