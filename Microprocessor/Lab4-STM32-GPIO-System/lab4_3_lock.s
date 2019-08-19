	.syntax unified
	.cpu cortex-m4
	.thumb
.data
    X:    .word 1280
    Y:    .word 780
    W:    .word 640
    Z:    .word 780
    test: .byte 0x9 //4'b1001
	password: .byte 0x9 //4'b1001
    status: .byte 0
 .text
 	.global main

    .equ RCC_AHB2ENR, 0x4002104C //clock enable register addr
    .equ GPIOA_MODER, 0x48000000 //port A mem start addr
 	.equ GPIOB_MODER, 0x48000400 //port B mem start addr
 	.equ GPIOB_ODR, 0x48000414 //(read) port B output, 0x14(offset)
 	.equ GPIOA_IDR, 0x48000010 //(write) port A input, 0x14(offset)

/*
 don't change:
 r4: times of blink
 */
 main:

 bl GPIO_init

 Loop:
    bl poll_user
    bl cmp_DIP
    bl DisplayLED

    // set status as current IDR
    ldr r0,=GPIOA_IDR
    ldr r2,[r0]
    lsr r2,#5
    and r2,#15 // 4'b1111
    ldr r0,=status
    str r2,[r0]
 b Loop

 L: b L

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

poll_user:
    // read DIP
    ldr r0,=GPIOA_IDR
    ldr r1,[r0]
    lsr r1,#5 // r1: IDR

    // cmp temp & last
    ldr r0,=status
    ldr r2,[r0] // r2: last status

    // pushed?
    and r1,#15
    eor r1,r2

    cmp r1,#0
    beq poll_user
bx lr

cmp_DIP:
    // read DIP
    ldr r0,=GPIOA_IDR
    ldr r1,[r0]
    lsr r1,#5 // r1: IDR

    // cmp PW & DIP
    ldr r0,=password
    ldr r2,[r0] // r2: PW

    // same?
    and r1,#15
    and r2,#15
    eor r2,#15
    eor r1,r2

    // set blink times
    cmp r1,#0
    IT eq
    moveq r4,#3
    IT ne
    movne r4,#1
bx lr

DisplayLED: // r4: times of blink
    ldr r1,=GPIOB_ODR
    blink:
    	// light
        movs r0,#0
        strh r0,[r1]

        // delay
        ldr r0,=X
    		ldr r5,[r0]
    		ldr r2,=Y
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

        // delay 0.5 sec
        ldr r0,=W
    		ldr r5,[r0]
    		ldr r2,=Z
    		LLL1:
        	ldr r3,[r2]
        	LLL2:
            subs r3,#1
        	bne LLL2
        	subs r5,#1
   	 	bne LLL1

    sub r4,#1
    cmp r4,#0
    bne blink
bx lr
