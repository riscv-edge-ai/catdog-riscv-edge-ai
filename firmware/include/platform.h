#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>

#define IMEM_BASE       0x00000000
#define DMEM_BASE       0x00040000
#define GPIO_BASE       0x00080000
#define ACCEL_BASE      0x000C0000
#define UART_BASE       0x000E0000
#define SRAM_BASE       0x00100000
#define CLINT_BASE      0x02000000

#define ACC_INPUT       (*(volatile int16_t*)(ACCEL_BASE + 0x00))
#define ACC_RESULT      (*(volatile int16_t*)(ACCEL_BASE + 0x04))
#define ACC_FUNC        (*(volatile uint32_t*)(ACCEL_BASE + 0x08))
#define ACC_STATUS      (*(volatile uint32_t*)(ACCEL_BASE + 0x0C))
#define ACC_COUNTER     (*(volatile uint32_t*)(ACCEL_BASE + 0x10))
#define ACC_SRC_ADDR    (*(volatile uint32_t*)(ACCEL_BASE + 0x14))
#define ACC_DST_ADDR    (*(volatile uint32_t*)(ACCEL_BASE + 0x18))
#define ACC_LEN         (*(volatile uint32_t*)(ACCEL_BASE + 0x1C))
#define ACC_CTRL        (*(volatile uint32_t*)(ACCEL_BASE + 0x20))
#define ACC_CTRL_START  0x1
#define ACC_CONV_IN     (*(volatile uint32_t*)(ACCEL_BASE + 0x24))
#define ACC_CONV_K      (*(volatile uint32_t*)(ACCEL_BASE + 0x28))
#define ACC_CONV_BIAS   (*(volatile uint32_t*)(ACCEL_BASE + 0x2C))
#define ACC_CONV_OUT    (*(volatile uint32_t*)(ACCEL_BASE + 0x30))
#define ACC_CONV_CHIN   (*(volatile uint32_t*)(ACCEL_BASE + 0x34))
#define ACC_CONV_H      (*(volatile uint32_t*)(ACCEL_BASE + 0x38))
#define ACC_CONV_W      (*(volatile uint32_t*)(ACCEL_BASE + 0x3C))
#define ACC_CONV_CHOUT  (*(volatile uint32_t*)(ACCEL_BASE + 0x40))
#define ACC_CONV_CTRL   (*(volatile uint32_t*)(ACCEL_BASE + 0x44))
#define ACC_CONV_STATUS (*(volatile uint32_t*)(ACCEL_BASE + 0x48))
#define ACC_MODEL_CYCLES (*(volatile uint32_t*)(ACCEL_BASE + 0x4C))
#define ACC_CONV_START  0x1

#define UART_TX_DATA    (*(volatile char*)(UART_BASE + 0x00))
#define UART_STATUS     (*(volatile uint32_t*)(UART_BASE + 0x04))

#define Q8_8_FRAC_BITS  8
#define Q8_8_SCALE      (1 << Q8_8_FRAC_BITS)
#define FLOAT_TO_Q8(x)  ((int16_t)((x) * Q8_8_SCALE))
#define Q8_TO_FLOAT(q)  ((float)(q) / Q8_8_SCALE)
#define Q8_MUL(a, b)    ((int16_t)(((int32_t)(a) * (b)) >> Q8_8_FRAC_BITS))
#define Q8_RELU(x)      ((x) > 0 ? (x) : 0)

#define ACCEL_RELU      0
#define ACCEL_SIGMOID   1
#define ACCEL_TANH      2

static inline int16_t bram_lut(int16_t x, uint32_t func) {
    ACC_FUNC = func;
    ACC_INPUT = x;
    while (!(ACC_STATUS & 1)) {
    }
    return ACC_RESULT;
}

static inline void bram_lut_batch(int16_t *data, uint32_t count, uint32_t func) {
    if (count == 0) return;
    ACC_FUNC = func;
    ACC_SRC_ADDR = (uint32_t)data;
    ACC_DST_ADDR = (uint32_t)data;
    ACC_LEN = count;
    ACC_STATUS = 0;
    ACC_CTRL = ACC_CTRL_START;
    while (!(ACC_STATUS & 1)) {
    }
}

#ifdef RENODE_ACCEL
static inline void conv2d_accel(const int16_t *input, int ch_in, int h, int w,
                                const int16_t *kernel, const int16_t *bias, int ch_out,
                                int16_t *output) {
    ACC_CONV_IN = (uint32_t)input;
    ACC_CONV_K = (uint32_t)kernel;
    ACC_CONV_BIAS = (uint32_t)bias;
    ACC_CONV_OUT = (uint32_t)output;
    ACC_CONV_CHIN = (uint32_t)ch_in;
    ACC_CONV_H = (uint32_t)h;
    ACC_CONV_W = (uint32_t)w;
    ACC_CONV_CHOUT = (uint32_t)ch_out;
    ACC_CONV_STATUS = 0;
    ACC_CONV_CTRL = ACC_CONV_START;
    while (!(ACC_CONV_STATUS & 1)) {
    }
}
#endif

static inline void accel_reset_modeled_cycles(void) {
    ACC_MODEL_CYCLES = 0;
}

static inline uint32_t accel_get_modeled_cycles(void) {
    return ACC_MODEL_CYCLES;
}

static inline uint64_t get_cycles(void) {
    uint32_t lo, hi;
    do {
        hi = *(volatile uint32_t*)(CLINT_BASE + 0x4);
        lo = *(volatile uint32_t*)(CLINT_BASE + 0x0);
    } while (hi != *(volatile uint32_t*)(CLINT_BASE + 0x4));
    return ((uint64_t)hi << 32) | lo;
}

static inline uint32_t get_cycles32(void) {
    return *(volatile uint32_t*)(CLINT_BASE + 0x0);
}

#endif
