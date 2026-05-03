PREFIX ?= riscv32-unknown-elf-
CC := $(PREFIX)gcc
OBJCOPY := $(PREFIX)objcopy
SIZE := $(PREFIX)size

CFLAGS := -march=rv32im -mabi=ilp32 -Os -Wall -Wextra -ffreestanding -nostdlib -Iinclude -Ilut -DRENODE_ACCEL
CFLAGS += $(EXTRA_CFLAGS)
LDFLAGS := -Tlink.ld -nostdlib -Wl,--gc-sections -lgcc

SRCS := src/start.S src/uart.c src/nn_ops.c src/main.c
OBJS := $(SRCS:.c=.o)
OBJS := $(OBJS:.S=.o)

TARGET := catdog_inference.elf

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LDFLAGS) -o $@
	$(SIZE) $@

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c $< -o $@

src/%.o: src/%.S
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean
