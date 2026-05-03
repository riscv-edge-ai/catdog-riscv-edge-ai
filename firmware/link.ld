ENTRY(_start)

MEMORY
{
    IMEM (rx)  : ORIGIN = 0x00000000, LENGTH = 0x00010000
    DMEM (rw)  : ORIGIN = 0x00040000, LENGTH = 0x00010000
    SRAM (rw)  : ORIGIN = 0x00100000, LENGTH = 0x00400000
}

SECTIONS
{
    .text : {
        KEEP(*(.init))
        *(.text*)
        *(.rodata*)
    } > IMEM

    .data : {
        __data_start = .;
        *(.data*)
        __data_end = .;
    } > DMEM

    .bss (NOLOAD) : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > DMEM

    .stack (NOLOAD) : {
        . = ALIGN(16);
    } > SRAM

    __stack_top = ORIGIN(SRAM) + LENGTH(SRAM);
}
