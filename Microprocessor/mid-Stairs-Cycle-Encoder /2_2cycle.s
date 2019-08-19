	.syntax unified
	.cpu cortex-m4
	.thumb
 .text
 	.global main
    .equ X, 1000000
    .equ Y, 1
 main:
 	bl Init_DWT
 	bl Delay
 L:
 	b L
 Init_DWT:
 	ldr r0, =0xE0000FB0
 	ldr r1, =0xC5ACCE55
 	str r1, [r0]

 	ldr r0, =0xE0000E80
 	ldr r1, [r0]
 	orr r1, 0x4
 	str r1, [r0]

 	ldr r0, =0xE0001000
 	ldr r1, [r0]
 	orr r1, 0x1
 	str r1, [r0]

 	ldr r0, =0xE0001004
 	mov r1, 0
 	str r1, [r0]
 	bx lr
 Delay:
 	ldr r1, =X
 L1:
 	ldr r2, =Y
 L2:
 	subs r2, #1
 	bne L2
 	subs r1, #1
 	bne L1
 	bx lr

