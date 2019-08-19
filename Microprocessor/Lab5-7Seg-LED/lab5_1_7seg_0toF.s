	.syntax unified
	.cpu cortex-m4
	.thumb
 .data
	arr: .byte 0x7E,0x30,0x6D,0x79,0x33,0x5B,0x5F,0x70,0x7F,0x7B,0x77,0x1F,0x4E,0x3D,0x4F,0x47 //TODO: put 0 to F 7-Seg LED pattern here
	X:	.word 1280
	Y:	.word 780
 .text
	.global main
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

 main:
	bl GPIO_init
	bl max7219_init
 loop:
	bl Display0toF
	b loop
 L: b L


 GPIO_init:
	//TODO: Initialize three GPIO pins as output for max7219 DIN, CS and CLK
	//enable AHB2 clock
 	movs r0,#0x2 //open port B
 	ldr r1,=RCC_AHB2ENR
 	str r0, [r1]

 	//set PB3 as output mode
 	movs r0,#0x540 //output pin3,4,5
 	ldr r1,=GPIOB_MODER
 	ldr r2,[r1] //r2:original things in GPIOB_MODER
 	and r2,#0xFFFFF03F //reset pin3,4,5
 	orrs r2,r2,r0
 	str r2,[r1]
 bx lr

 Display0toF:
	//TODO: Display 0 to F at first digit on 7-SEG LED. Display one per second.
 push {lr}
	mov r3,#0

	go:
		ldr r4,=arr
		ldrb r2,[r4,r3] //r2: arr[j] value
		mov r0,#1
		mov r1,r2
		bl MAX7219Send
		bl Delay
		add r3,#1
	cmp r3,#16
	blt go
 pop {pc}

 MAX7219Send:
	//input parameter: r0 is ADDRESS , r1 is DATA
	//TODO: Use this function to send a message to max7219
	push {r3}
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
	pop {r3}
 bx lr

 max7219_init:
	//TODO: Initialize max7219 registers
 push {lr}
	ldr r0, =#DECODE_MODE
	ldr r1, =#0x0
	bl MAX7219Send
	ldr r0, =#DISPLAY_TEST
	ldr r1, =#0x0
	bl MAX7219Send
	ldr r0, =#SCAN_LIMIT
	ldr r1, =0x0 //1st digit
	bl MAX7219Send
	ldr r0, =#INTENSITY
	ldr r1, =#0xA //lightness
	bl MAX7219Send
	ldr r0, =#SHUTDOWN
	ldr r1, =#0x1 //bright
	bl MAX7219Send
 pop {pc}

 Delay:
	ldr r5,=X
    ldr r6,[r5]
    ldr r7,=Y
    LL1:
 	    ldr r8,[r7]
        LL2:
 	    subs r8,#1
 	    bne LL2
 	    subs r6,#1
 	bne LL1
 bx lr
