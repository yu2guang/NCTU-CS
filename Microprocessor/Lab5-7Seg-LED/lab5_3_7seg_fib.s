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
	.equ GPIOC_MODER, 0x48000800//button
 	.equ GPIOC_IDR, 0x48000810
 add_N:
 	// N++
	ldr r0,=N
	ldr r1,[r0]
	add r1,r1,#1
	str r1,[r0]
	// flag = 1
	ldr r0,=flag
	mov r1,#1
	str r1,[r0]

	b change

 Delay:    //TODO: Write a delay 1 sec function

 	ldr r5,=X
 	ldr r9,[r5]
 	ldr r7,=Y
 	LL1:
 		ldr r8,[r7]
 		LL2:
 		subs r8,#1
 		bne LL2

 		ldr r5,=GPIOC_IDR
 	    ldr r0,[r5] //r0:IDR value
		lsr r0,#13 //PC13
		cmp r0,#1
		beq delay_back

 		subs r9,#1
 	bne LL1

 	ldr r8,=flag //flag=0
 	mov r9,#0
 	str r9,[r8]

 	ldr r5,=GPIOC_IDR
 	ldr r0,[r5] //r0:IDR value
	lsr r0,#13 //PC13
	cmp r0,#0
 	beq zero
 delay_back:
 	ldr r8,=flag //flag=0
 	mov r9,#0
 	str r9,[r8]
 	mov pc,r6


 fib:
 	//TODO
 	movs r1,#0
 	movs r4,#1 //r4: result
 loop2:
 	mov r2,r4 //r2: 2nd
 	add r4,r1,r4 //r4: 3rd
 	ldr r5,=#100000000
 	cmp r4,r5
 	bge over
 	mov r1,r2 //r1: 1st
 	subs r0,#1
 	cmp r0,#1
 	bne loop2
	ldr r0,=fib_result
	str r4,[r0]
 	bl cut
 over:
 	mov r3,#2
 	ldr r4,=arr
 	mov r0,#1
 	str r0,[r4,#0]
 	mov r0,#10
 	str r0,[r4,#1]
 	bl send_fib
 zero:
 	ldr r4,=N
 	mov r0,#0
 	str r0,[r4]
 	mov r3,#1
 	ldr r4,=arr
 	mov r0,#0
 	str r0,[r4,#0]
 	bl send_fib
 one:
 	ldr r4,=N
 	mov r0,#1
 	str r0,[r4]
 	mov r3,#1
 	ldr r4,=arr
 	mov r0,#1
 	str r0,[r4,#0]
 	bl send_fib

 /* main */
 main:
 	// initialize
	bl GPIO_init
	bl max7219_init

 change:
	ldr r4,=N
 	ldr r0,[r4] // r0: button input N

	cmp r0,#0
	beq zero
	cmp r0,#1
	beq one
 	bl fib

 cut:
 	ldr r1,=fib_result
	ldr r0,[r1] //r0: fib result
	mov r3,#0
	ldr r4,=arr //arr start addr

	cut_loop:
		mov r1,#10 //r1: divisor
		udiv r2,r0,r1 //r2: r0/10
		mls r0,r1,r2,r0 //r0=r0-r2*r1, r0: result%10
		str r0,[r4,r3]
		add r3,#1 //r3: number of digit
		mov r0,r2 //r0=r0/10
		cmp r0,#0
		bne cut_loop
 send_fib:
	ldr r0, =n_of_digit
	str r3,[r0]
	ldr r0, =#SCAN_LIMIT
	sub r3,r3,#1
	mov r1,r3
	bl MAX7219Send

 loop:
	bl Display
	b loop
 L: b L
 /* main */


 GPIO_init:
	//TODO: Initialize three GPIO pins as output for max7219 DIN, CS and CLK
	//enable AHB2 clock
 	movs r0,#0x6 //open port B&C
 	ldr r1,=RCC_AHB2ENR
 	str r0, [r1]

 	//set PB3 as output mode
 	movs r0,#0x540 //output pin3,4,5
 	ldr r1,=GPIOB_MODER
 	ldr r2,[r1] //r2:original things in GPIOB_MODER
 	and r2,#0xFFFFF03F //reset pin3,4,5
 	orrs r2,r2,r0
 	str r2,[r1]
 	//set PC13 as input mode
 	ldr r1,=GPIOC_MODER
 	ldr r0,[r1]
 	and r0,#0xF3FFFFFF
 	str r0,[r1]
 bx lr

 Display:

 mov r6,lr

	ldr r3,=n_of_digit
	ldr r7,[r3]
	mov r3,#0
	mov r5,#1
	ldr r4,=arr

	go:
		ldrb r2,[r4,r3] //r2: arr[j] value
		mov r0,r5
		mov r1,r2
		bl MAX7219Send
		add r3,#1
		add r5,#1
	cmp r3,r7
	blt go

    ldr r0,=flag
    ldr r1,[r0]
    cmp r1,#1
    beq Delay

	ldr r0,=N
	ldr r1,[r0]
	cmp r1,#0
	bne N_ge_1
    little_loop:
	ldr r5,=GPIOC_IDR
 	ldr r0,[r5] //r0:IDR value
	lsr r0,#13 //PC13
	cmp r0,#0
	beq little_loop

 N_ge_1:
    bl determine//button!!!!!!!!!!!!!!!!!!!
 mov pc,r6

 MAX7219Send:
	//input parameter: r0 is ADDRESS , r1 is DATA
	//TODO: Use this function to send a message to max7219
	mov r10,lr
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


 mov pc,r10

 max7219_init:
	//TODO: Initialize max7219 registers
 push {lr}
	ldr r0, =#DECODE_MODE
	ldr r1, =#0xFF
	bl MAX7219Send
	ldr r0, =#DISPLAY_TEST
	ldr r1, =#0x0
	bl MAX7219Send

	ldr r0, =#INTENSITY
	ldr r1, =#0xA //lightness
	bl MAX7219Send
	ldr r0, =#SHUTDOWN
	ldr r1, =#0x1 //bright
	bl MAX7219Send
 pop {pc}

 determine:
	push {lr}
	push {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10}
	movs r6,#0
	movs r7,#0
	ldr r5,=GPIOC_IDR
	mov r8,#0

	stable_loop: //r6:count
	ldr r9,=#100
 	cmp r8,r9
 	beq back

	ldr r0,[r5] //r0:IDR value
	lsr r0,#13
	cmp r0,#0
	IT eq
	addeq r6,r6,#1
	cmp r6,#4 //r7:count=4->r7=1
	IT eq
	moveq r7,#1
	cmp r7,#1
	beq add_N
	add r8,r8,#1

	b stable_loop

back:
	pop {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10}
 	pop {pc}



	//b change
 /*stable_loop: //r6:count N
 	ldr r9,=#1000010
 	cmp r8,r9
 	beq back
	ldr r0,[r5] //r0:IDR value
	lsr r0,#13 //PC13
	cmp r0,#0
	IT eq
	addeq r6,r6,#1
	ldr r9,=#1000000
	cmp r6,r9 //r7:count=4->r7=1
	bgt zero
	cmp r6,#4 //r7:count=4->r7=1
	IT eq
	moveq r7,#1
	and r1,r0,r7
	cmp r1,#1
	ITTT eq
	moveq r6,#0
	moveq r7,#0
	beq add_N
	cmp r0,#1
	IT eq
	moveq r6,#0
	add r8,r8,#1
	b stable_loop

 add_N:
	ldr r0,=N
	ldr r1,[r0]
	add r1,r1,#1
	str r1,[r0]
	pop {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10}
 	pop {pc}
	b change

 back:
 	pop {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10}
 	pop {pc}*/

