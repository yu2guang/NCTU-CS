	.syntax unified
	.cpu cortex-m4
	.thumb
 .data
	leds: .byte 0
	X:	.word 1280
	Y:	.word 780
 .text
.global main

 	.equ RCC_AHB2ENR, 0x4002104C //clock enable register addr
 	.equ GPIOB_MODER, 0x48000400 //port B mem start addr
 	.equ GPIOB_ODR, 0x48000414 //(read) port B output, 0x14(offset)
 main:
 	ldr r9, =leds
 	ldrb r3,[r9]
 	movs r3,#1
    strb r3, [r9]

	BL   GPIO_init
    BL Loop


 L: B L

 Loop: //TODO: Write the display pattern into leds variable
 	//r9:leds's addr
  	BL DisplayLED
  	BL   Delay
  	ldrb r3,[r9]
  	lsl r3,r3,#1
  	add r3,r3,#1
  	strb r3,[r9]
  	BL DisplayLED
  	BL   Delay
  	while1:
  		ldrb r3,[r9]
  		lsl r3,r3,#1
  		strb r3,[r9]
  		cmp r3,#24
  		beq shift_r
  		BL DisplayLED
  		BL   Delay
  	b while1

  	shift_r: //is 24
  		BL DisplayLED
  		BL   Delay
  		while2:
  			ldrb r3,[r9]
  			lsr r3,r3,#1
  			strb r3,[r9]
  			cmp r3,#1
  			beq Loop
  			BL DisplayLED
  			BL   Delay
  		b while2


  	B Loop
 GPIO_init:   //TODO: Initial LED GPIO pins as output

 	//enable AHB2 clock
 	movs r0,#0x2 //open port B
 	ldr r1,=RCC_AHB2ENR
 	str r0, [r1]

 	//set PB3 as output mode
 	movs r0,#0x1540 //output pin3,4,5,6
 	ldr r1,=GPIOB_MODER
 	ldr r2,[r1] //r2:original things in GPIOB_MODER
 	and r2,#0xFFFFC03F //reset pin3,4,5,6
 	orrs r2,r2,r0
 	str r2,[r1]

 	bx lr
 DisplayLED: //TODO: Display LED by leds

	ldrb r3,[r9]
 	movs r4,r3
 	eor r4,r4,#15
 	lsl r4,#3

 	ldr r1,=GPIOB_ODR
 	L1:
 		movs r0,r4
 		strh r0,[r1]

 	bx lr
 Delay:    //TODO: Write a delay 1 sec function

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
