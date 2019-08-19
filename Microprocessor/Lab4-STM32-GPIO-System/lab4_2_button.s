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
 	.equ GPIOC_MODER, 0x48000800//button
 	.equ GPIOC_IDR, 0x48000810

/*
	don't change:
	r3: leds
	r5: GPIOC_IDR's addr
*/

 main:
	movs r3,#1
	BL   GPIO_init
 	ldr r9, =leds
 	ldrb r3,[r9]
 	movs r3,#1
    strb r3, [r9]
 	BL Loop

 L: B L

 Loop: //TODO: Write the display pattern into leds variable
 	//r3:leds's value
  	BL DisplayLED
  	BL   Delay
  	ldr r1,[r5]
  	cmp r1,#0
  	IT eq
  	BLEQ   determine
  	ldrb r3,[r9]
  	lsl r3,r3,#1
  	add r3,r3,#1
  	strb r3,[r9]
  	BL DisplayLED
  	BL   Delay
  	ldr r1,[r5]
  	cmp r1,#0
  	IT eq
  	BLEQ   determine

  	while1:
  		ldrb r3,[r9]
  		lsl r3,r3,#1
  		strb r3,[r9]
  		cmp r3,#24
  		beq shift_r
  		BL DisplayLED
  		BL   Delay
  		ldr r1,[r5]
  		cmp r1,#0
  		IT eq
  		BLEQ   determine
  	b while1

  	shift_r: //is 24
  		BL DisplayLED
  		BL   Delay
  		ldr r1,[r5]
  		cmp r1,#0
  		IT eq
  		BLEQ   determine
  		while2:
  		ldrb r3,[r9]
  		lsr r3,r3,#1
  		strb r3,[r9]
  		cmp r3,#1
  		beq Loop
  		BL DisplayLED
  		BL   Delay
  		ldr r1,[r5]
  		cmp r1,#0
  		IT eq
  		BLEQ   determine
  		b while2


  	B Loop
 GPIO_init:   //TODO: Initial LED GPIO pins as output

 	//enable AHB2 clock
 	movs r0,#0x6 //open port B&C
 	ldr r1,=RCC_AHB2ENR
 	str r0, [r1]

 	//set PB3 as output mode
 	movs r0,#0x1540 //output pin3,4,5,6
 	ldr r1,=GPIOB_MODER
 	ldr r2,[r1] //r2:original things in GPIOB_MODER
 	and r2,#0xFFFFC03F //reset pin3,4,5,6
 	orrs r2,r2,r0
 	str r2,[r1]
 	//set PC13 as input mode
 	ldr r1,=GPIOC_MODER
 	ldr r0,[r1]
 	and r0,#0xF3FFFFFF
 	str r0,[r1]
 	//set data register addr
 	ldr r5,=GPIOC_IDR//r5:GPIOC_IDR's addr
 	ldr r1,[r5]

 	BX LR
 DisplayLED: //TODO: Display LED by leds

	ldrb r3,[r9]
 	movs r4,r3
 	eor r4,r4,#15
 	lsl r4,#3

 	ldr r1,=GPIOB_ODR
 	L1:
 		movs r0,r4
 		strh r0,[r1]

 	BX LR
 Delay:    //TODO: Write a delay 1 sec function

 	ldr r0,=X
 	ldr r1,[r0]
 	ldr r2,=Y
 	LL1:
 		ldr r8,[r2]
 		LL2:
 			subs r8,#1
 		bne LL2
 		subs r1,#1
 	bne LL1
 	bx lr

//decide if someone push the button!!!!!!!!!!
determine:
	movs r6,#0
	movs r7,#0
stable_loop: //r6:count
	ldr r0,[r5] //r0:IDR value
	lsr r0,#13
	cmp r0,#0
	IT eq
	addeq r6,r6,#1
	cmp r6,#4 //r7:count=7->r7=1
	IT eq
	moveq r7,#1
	and r1,r0,r7
	cmp r1,#1
	ITTT eq
	moveq r6,#0
	moveq r7,#0
	beq stop_loop
	cmp r0,#1
	IT eq
	moveq r6,#0
	b stable_loop

 stop_loop:
	ldr r0,[r5] //r0:IDR value
	lsr r0,#13
	cmp r0,#0
	IT eq
	addeq r6,r6,#1
	cmp r6,#4 //r7:count=7->r7=1
	IT eq
	moveq r7,#1
	and r1,r0,r7
	cmp r1,#1
	ITT eq
	moveq r6,#0
	moveq r7,#0
	beq back
	cmp r0,#1
	IT eq
	moveq r6,#0
	b stop_loop

 back:
	bx lr





