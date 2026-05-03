#include "platform.h"
#include "uart.h"

void uart_putchar(char c) {
    // LiteX UART: status bit0 = TX full. Wait until not full.
    while (UART_STATUS & 0x1) {
    }
    UART_TX_DATA = c;
}

void uart_puts(const char *s) {
    while (*s) {
        uart_putchar(*s++);
    }
}
