	.syntax unified
	.cpu cortex-m4
	.thumb
 .data
	X:    .word 1280
	Y:    .word 65
	W:    .word 2560
	Z:    .word 780
	status: .byte 0
	decode_LED: .byte 0
 .text
	.global main
	.equ RCC_AHB2ENR, 0x4002104C //clock enable register addr
	.equ GPIOA_MODER, 0x48000000 //port A mem start addr
	.equ GPIOB_MODER, 0x48000400 //port B mem start addr
	.equ GPIOB_ODR, 0x48000414 //(read) port B output, 0x14(offset)
	.equ GPIOA_IDR, 0x48000010 //(write) port A input, 0x14(offset)

 GPIO_init:
	// enable AHB2 clock
	movs r0,#0x3 //open port A & B
	ldr r1,=RCC_AHB2ENR
	str r0, [r1]

	// set PB mode
	mov r0,#0x1540 // output: PB3,4,5,6->LED
	ldr r1,=GPIOB_MODER
	ldr r2,[r1] // r2: original things in GPIOB_MODER
	and r2,#0xFFFFC03F // reset pin3,4,5,6
	orrs r2,r2,r0
	str r2,[r1]

	// turn off LED
	ldr r1,=GPIOB_ODR
	movs r0,#(15<<3)
	strh r0,[r1]

	// set PA mode
	mov r0,#0//input: PA 5,6,7,8->DIP switch
	ldr r1,=GPIOA_MODER
	ldr r2,[r1] // r2: original things in GPIOA_MODER
	and r2,#0xFFFC03FF // reset pin5,6,7,8
	orrs r2,r2,r0
	str r2,[r1]

	// set status as current IDR
	ldr r0,=status
	mov r2,#15
	str r2,[r0]
	bx lr

 main:
	bl GPIO_init
	bl poll_user
	b cmp_DIP

 poll_user:
	push {r0,r1,r2,r3,r4,r5}
	// read DIP
	ldr r0,=GPIOA_IDR
	ldr r1,[r0]
	lsr r1,#5 // r1: IDR
	and r1,#15

	// read status
	ldr r0,=status
	ldr r2,[r0] // r2: last status
	and r2,#15


	// pushed?
	eor r4,r1,r2
	cmp r4,#0
	bne change
	pop {r0,r1,r2,r3,r4,r5}
	bx lr

 change:
	// set status as current IDR
	str r1,[r0]
	b cmp_DIP


 cmp_DIP:
	// read DIP
	ldr r0,=GPIOA_IDR
	ldr r1,[r0]
	lsr r1,#5 // r1: IDR
	and r1,#15
	eor r1,#15

	mov r0,#4
	mov r3,#0
 upside_down:
	mov r2,r1
	and r2,#1
	lsl r3,#1
	add r3,r2
	lsr r1,#1
	subs r0,#1
 bne upside_down

	// LED light
	ldr r4,=decode_LED
	sub r3,#3 // decode

	mov r1,r3
	mov r0,#4
	mov r3,#0
 upside_down2:
	mov r2,r1
	and r2,#1
	lsl r3,#1
	add r3,r2
	lsr r1,#1
	subs r0,#1
 bne upside_down2

	eor r3,#15
	lsl r3,#3
	str r3,[r4]

 DisplayLED:
	ldr r1,=GPIOB_ODR
	blink:
	// light
	ldr r0,=decode_LED
	ldr r2,[r0]
	strh r2,[r1]

	// delay 4 sec
	ldr r0,=W
	ldr r5,[r0]
	ldr r2,=Z
 	LL1:
	ldr r3,[r2]
	LL2:
	subs r3,#1
	bne LL2
	subs r5,#1
	bne LL1

	// dark
	movs r0,#(15<<3)
	strh r0,[r1]

	// delay 1 sec
	ldr r0,=X
	ldr r5,[r0]
	ldr r2,=Y
	LLL1:
	ldr r3,[r2]
	LLL2:
	bl poll_user
	subs r3,#1
	bne LLL2
	subs r5,#1
	bne LLL1
 b DisplayLED
