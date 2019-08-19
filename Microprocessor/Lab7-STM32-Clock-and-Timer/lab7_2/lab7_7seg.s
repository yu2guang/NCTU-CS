	.syntax unified
	.cpu cortex-m4
	.thumb
 .data
	arr: .byte 0x0,0x0,0x0,0x0,0x0,0x0,0x0,0x0
	fib_result:	.word 0
	n_of_digit: .word 0
	N: 	.word 0
	X:	.word 1280
	Y:	.word 780
	flag: .word 0 //if the button was pushed=1
 .text
	.global max7219_init
	.global max7219Send
	.equ RCC_AHB2ENR, 0x4002104C //clock enable register addr
 	.equ GPIOB_MODER, 0x48000400 //port B mem start addr
 	.equ GPIOB_ODR, 0x48000414 //(read) port B output, 0x14(offset)
 	.equ GPIO_BSRR_OFFSET, 0x18 //bit set/reset reg
 	.equ GPIO_BRR_OFFSET, 0x28 //bit reset reg
 	.equ DATA, 0x8 //PB3
 	.equ LOAD, 0x10 //PB4
 	.equ CLOCK, 0x20 //PB5
 	.equ DECODE_MODE, 0x09
 	.equ DISPLAY_TEST, 0x0F
	.equ SCAN_LIMIT, 0x0B
	.equ INTENSITY, 0x0A
	.equ SHUTDOWN, 0x0C
	.equ GPIOC_MODER, 0x48000800//button
 	.equ GPIOC_IDR, 0x48000810

 max7219Send:
	//input parameter: r0 is ADDRESS , r1 is DATA
	//TODO: Use this function to send a message to max7219
 push {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10}
	lsl r0, r0, #8
	add r0, r0, r1
	ldr r1, =#GPIOB_MODER
	ldr r2, =#LOAD
	ldr r3, =#DATA
	ldr r4, =#CLOCK
	ldr r5, =#GPIO_BSRR_OFFSET
	ldr r6, =#GPIO_BRR_OFFSET
	mov r7, #16 //r7 = i

	//send msg to each bit
	send_loop:
		//set r8 i-th bit as 1, others 0
		mov r8, #1
		sub r9, r7, #1
		lsl r8, r8, r9 //r8 = mask

		str r4, [r1,r6] //CLK=0
		tst r0, r8 //ands
		beq bit_not_set //if above = 0, branch
		str r3, [r1,r5] //DATA = 1
		b if_done
		bit_not_set:
		str r3, [r1,r6] //DATA = 0
		if_done:
		str r4, [r1,r5] //CLK=1, read data

	subs r7,r7,#1
	bgt send_loop
	str r2, [r1,r6] //LOAD = 0
	str r2, [r1,r5] //LOAD = 1

	pop {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10}
 bx lr

 max7219_init:
	//TODO: Initialize max7219 registers
 push {lr}
 push {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10}
	ldr r0, =#DECODE_MODE
	ldr r1, =#0x0
	bl max7219Send
	ldr r0, =#DISPLAY_TEST
	ldr r1, =#0x0
	bl max7219Send

	ldr r0, =#INTENSITY
	ldr r1, =#0xA //lightness
	bl max7219Send
	ldr r0, =#SHUTDOWN
	ldr r1, =#0x1 //bright
	bl max7219Send

 pop {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10}
 pop {pc}

