#include "platform.h"
#include "uart.h"
#include "nn_ops.h"
#include "weights.h"
#include "eval_dataset_meta.h"

// Model dimensions
#define INPUT_CH    1
#define INPUT_H     32
#define INPUT_W     32

// After conv1 (same) + pool: 8 x 16 x 16
#define C1_OUT      8
#define H1_PRE      32
#define W1_PRE      32
#define H1          16
#define W1          16

// After conv2 (same) + pool: 16 x 8 x 8
#define C2_OUT      16
#define H2_PRE      16
#define W2_PRE      16
#define H2          8
#define W2          8

// After conv3 (same) + pool: 32 x 4 x 4
#define C3_OUT      32
#define H3_PRE      8
#define W3_PRE      8
#define H3          4
#define W3          4

#define FC_IN       512
#define FC_OUT      2

#define EVAL_IMAGE_PIXELS 1024
#define EVAL_DATASET_BASE SRAM_BASE
#define RUN_IMAGE_COUNT EVAL_DATASET_COUNT
#define VERBOSE_STAGE_LOGS (RUN_IMAGE_COUNT <= 20)
#define VERBOSE_IMAGE_LOGS (RUN_IMAGE_COUNT <= 100)
#ifdef SKIP_SW_BENCHMARK
#define RUN_SW_BENCHMARK 0
#else
#define RUN_SW_BENCHMARK 1
#endif

// Buffers
static int16_t layer1_out[C1_OUT * H1_PRE * W1_PRE];
static int16_t layer2_out[C2_OUT * H2_PRE * W2_PRE];
static int16_t layer3_out[C3_OUT * H3_PRE * W3_PRE];
static int16_t fc_in[FC_IN];
static int16_t fc_out[FC_OUT];
static int profile_enabled = 0;

static void print_uint32(uint32_t v);
static uint32_t get_profile_cycles(void);
static const int16_t *get_eval_image(uint32_t idx);

static void print_stage_cycles(const char *label, uint32_t cycles) {
    uart_puts("  ");
    uart_puts(label);
    uart_puts(": ");
    print_uint32(cycles);
    uart_puts("\r\n");
}

static uint32_t get_profile_cycles(void) {
    return get_cycles32();
}

static const int16_t *get_eval_image(uint32_t idx) {
    return (const int16_t *)(uintptr_t)(EVAL_DATASET_BASE + idx * EVAL_IMAGE_PIXELS * sizeof(int16_t));
}

void run_inference(const int16_t *input, int use_accel) {
    use_bram_accel = use_accel;
    use_conv_accel = use_accel;
    uint32_t t0 = 0;
    uint32_t t1 = 0;
    if (profile_enabled) {
        t0 = get_profile_cycles();
    }

    // Conv1 + ReLU + Pool
    if (profile_enabled) t0 = get_profile_cycles();
    conv2d_q8_same(input, INPUT_CH, INPUT_H, INPUT_W, conv1_weight, conv1_bias, C1_OUT, layer1_out);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("conv1", t1 - t0);
        t0 = t1;
    }
    if (use_accel) {
        use_bram_accel = 0;
    }
    apply_activation(layer1_out, C1_OUT * H1_PRE * W1_PRE, ACCEL_RELU);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("relu1", t1 - t0);
        t0 = t1;
    }
    maxpool2d_q8(layer1_out, C1_OUT, H1_PRE, W1_PRE, layer1_out);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("pool1", t1 - t0);
        t0 = t1;
    }

    // Conv2 + ReLU + Pool
    conv2d_q8_same(layer1_out, C1_OUT, H1, W1, conv2_weight, conv2_bias, C2_OUT, layer2_out);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("conv2", t1 - t0);
        t0 = t1;
    }
    if (use_accel) {
        use_bram_accel = 1;
    }
    apply_activation(layer2_out, C2_OUT * H2_PRE * W2_PRE, ACCEL_RELU);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("relu2", t1 - t0);
        t0 = t1;
    }
    maxpool2d_q8(layer2_out, C2_OUT, H2_PRE, W2_PRE, layer2_out);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("pool2", t1 - t0);
        t0 = t1;
    }

    // Conv3 + ReLU + Pool
    conv2d_q8_same(layer2_out, C2_OUT, H2, W2, conv3_weight, conv3_bias, C3_OUT, layer3_out);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("conv3", t1 - t0);
        t0 = t1;
    }
    apply_activation(layer3_out, C3_OUT * H3_PRE * W3_PRE, ACCEL_RELU);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("relu3", t1 - t0);
        t0 = t1;
    }
    maxpool2d_q8(layer3_out, C3_OUT, H3_PRE, W3_PRE, layer3_out);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("pool3", t1 - t0);
        t0 = t1;
    }

    // Flatten
    for (int i = 0; i < FC_IN; i++) {
        fc_in[i] = layer3_out[i];
    }
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("flatten", t1 - t0);
        t0 = t1;
    }

    // FC + sigmoid
    fc_q8(fc_in, FC_IN, fc_weight, FC_OUT, fc_bias, fc_out);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("fc", t1 - t0);
        t0 = t1;
    }
    apply_activation(fc_out, FC_OUT, ACCEL_SIGMOID);
    if (profile_enabled) {
        t1 = get_profile_cycles();
        print_stage_cycles("sigmoid", t1 - t0);
    }
}

static void print_uint32(uint32_t v) {
    char buf[32];
    int i = 0;
    if (v == 0) {
        uart_putchar('0');
        return;
    }
    while (v) {
        buf[i++] = '0' + (v % 10);
        v /= 10;
    }
    while (i--) uart_putchar(buf[i]);
}

static void print_fixed_3(uint64_t value_x1000) {
    // Prints value with 3 decimal places from integer scaled by 1000
    uint32_t int_part = (uint32_t)(value_x1000 / 1000);
    uint32_t frac = (uint32_t)(value_x1000 % 1000);
    print_uint32(int_part);
    uart_putchar('.');
    uart_putchar('0' + (frac / 100));
    uart_putchar('0' + ((frac / 10) % 10));
    uart_putchar('0' + (frac % 10));
}

static void print_latency_fps(uint32_t cycles) {
    // CPU is 25 MHz => 25,000,000 cycles per second
    const uint32_t cpu_hz = 25000000;
    // latency_ms_x1000 = (cycles * 1000 * 1000) / cpu_hz
    uint64_t latency_ms_x1000 = ((uint64_t)cycles * 1000000ULL) / cpu_hz;
    // fps_x1000 = (cpu_hz * 1000) / cycles
    uint64_t fps_x1000 = ((uint64_t)cpu_hz * 1000ULL) / (cycles ? cycles : 1);

    uart_puts("Simulated latency: ");
    print_fixed_3(latency_ms_x1000);
    uart_puts(" ms\r\n");
    uart_puts("Simulated FPS: ");
    print_fixed_3(fps_x1000);
    uart_puts("\r\n");
}

static void print_average_latency_fps(const char *label, uint64_t total_cycles, uint32_t count) {
    const uint32_t cpu_hz = 25000000;
    uint64_t avg_cycles = count ? (total_cycles / count) : 0;
    uint64_t latency_ms_x1000 = (avg_cycles * 1000000ULL) / cpu_hz;
    uint64_t fps_x1000 = ((uint64_t)cpu_hz * 1000ULL) / (avg_cycles ? avg_cycles : 1);

    uart_puts(label);
    uart_puts(" average latency: ");
    print_fixed_3(latency_ms_x1000);
    uart_puts(" ms\r\n");
    uart_puts(label);
    uart_puts(" average FPS: ");
    print_fixed_3(fps_x1000);
    uart_puts("\r\n");
}

int main() {
    uint64_t accel_cycles_total = 0;
    uint64_t sw_cycles_total = 0;
    uint64_t accel_modeled_total = 0;
    uint64_t bram_access_total = 0;
    uint32_t accel_correct = 0;
    uint32_t sw_correct = 0;

    uart_puts("\r\n=============================================\r\n");
    uart_puts(" ECP5 RISC-V SoC - Cat/Dog Classifier\r\n");
    uart_puts(" Renode-Modeled Accelerator Benchmark\r\n");
    uart_puts("=============================================\r\n\r\n");

    for (int run_idx = 0; run_idx < RUN_IMAGE_COUNT; run_idx++) {
        uint32_t img_idx = (uint32_t)run_idx;
        const int16_t *test_img = get_eval_image(img_idx);
        int expected = eval_expected_labels[img_idx];

        uart_puts("=== Image ");
        print_uint32((uint32_t)run_idx);
        uart_puts(" (src ");
        print_uint32((uint32_t)img_idx);
        uart_puts(") ===\r\n");

        uart_puts("--- Renode-Modeled Accelerator (simulated) ---\r\n");
        profile_enabled = VERBOSE_STAGE_LOGS;
        accel_reset_modeled_cycles();
        uint32_t bram_before = ACC_COUNTER;
        uint32_t start = get_profile_cycles();
        run_inference(test_img, 1);
        uint32_t end = get_profile_cycles();
        uint32_t accel_cycles = end - start;
        profile_enabled = 0;
        uint32_t bram_after = ACC_COUNTER;
        uint32_t modeled_cycles = accel_get_modeled_cycles();
        const char *accel_prediction = fc_out[0] > fc_out[1] ? "CAT" : "DOG";
        uint32_t bram_accesses = bram_after - bram_before;
        accel_cycles_total += accel_cycles;
        accel_modeled_total += modeled_cycles;
        bram_access_total += bram_accesses;
        if ((fc_out[0] > fc_out[1] ? 0 : 1) == expected) {
            accel_correct++;
        }
        uart_puts("[ACCEL_SIM] Cycles: ");
        print_uint32(accel_cycles);
        uart_puts("\r\n");
        if (VERBOSE_IMAGE_LOGS) {
            print_latency_fps(accel_cycles);
        }
        uart_puts("Modeled accelerator cycles: ");
        print_uint32(modeled_cycles);
        uart_puts("\r\n");
        uart_puts("Prediction: ");
        uart_puts(accel_prediction);
        uart_puts("\r\nExpected: ");
        uart_puts(expected == 0 ? "CAT" : "DOG");
        if (VERBOSE_IMAGE_LOGS) {
            uart_puts("\r\nBRAM accesses: ");
            print_uint32(bram_accesses);
        }
        uart_puts("\r\n\r\n");

#if RUN_SW_BENCHMARK
        uart_puts("--- Software Only (Renode timing) ---\r\n");
        profile_enabled = VERBOSE_STAGE_LOGS;
        accel_reset_modeled_cycles();
        start = get_profile_cycles();
        run_inference(test_img, 0);
        end = get_profile_cycles();
        uint32_t sw_cycles = end - start;
        profile_enabled = 0;
        const char *sw_prediction = fc_out[0] > fc_out[1] ? "CAT" : "DOG";
        sw_cycles_total += sw_cycles;
        if ((fc_out[0] > fc_out[1] ? 0 : 1) == expected) {
            sw_correct++;
        }
        uart_puts("[SW_ONLY]  Cycles: ");
        print_uint32(sw_cycles);
        uart_puts("\r\n");
        if (VERBOSE_IMAGE_LOGS) {
            print_latency_fps(sw_cycles);
        }
        uart_puts("Prediction: ");
        uart_puts(sw_prediction);
        uart_puts("\r\nExpected: ");
        uart_puts(expected == 0 ? "CAT" : "DOG");
        uart_puts("\r\n\r\n");
#endif
    }

    uart_puts("=== Average Benchmark Summary ===\r\n");
    uart_puts("Images: ");
    print_uint32((uint32_t)RUN_IMAGE_COUNT);
    uart_puts("\r\n");
    uart_puts("Average accelerator cycles: ");
    print_uint32((uint32_t)(accel_cycles_total / RUN_IMAGE_COUNT));
    uart_puts("\r\n");
#if RUN_SW_BENCHMARK
    uart_puts("Average software cycles: ");
    print_uint32((uint32_t)(sw_cycles_total / RUN_IMAGE_COUNT));
    uart_puts("\r\n");
    print_average_latency_fps("Software", sw_cycles_total, RUN_IMAGE_COUNT);
#endif
    print_average_latency_fps("Accelerator", accel_cycles_total, RUN_IMAGE_COUNT);
    uart_puts("Average modeled accelerator cycles: ");
    print_uint32((uint32_t)(accel_modeled_total / RUN_IMAGE_COUNT));
    uart_puts("\r\n");
    uart_puts("Average BRAM accesses: ");
    print_uint32((uint32_t)(bram_access_total / RUN_IMAGE_COUNT));
    uart_puts("\r\n");
    uart_puts("Accelerator accuracy: ");
    print_uint32(accel_correct);
    uart_puts("/");
    print_uint32((uint32_t)RUN_IMAGE_COUNT);
    uart_puts("\r\n");
#if RUN_SW_BENCHMARK
    uart_puts("Software accuracy: ");
    print_uint32(sw_correct);
    uart_puts("/");
    print_uint32((uint32_t)RUN_IMAGE_COUNT);
    uart_puts("\r\n");
#endif
    uart_puts("Done.\r\n");
    return 0;
}
