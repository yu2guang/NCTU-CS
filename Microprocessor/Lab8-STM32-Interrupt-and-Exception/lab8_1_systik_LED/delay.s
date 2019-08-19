	.syntax unified
	.cpu cortex-m4
	.thumb
 .data
	leds: .byte 0
	X:	.word 1280
	Y:	.word 780
 .text
.global Delay

 Delay:    //TODO: Write a delay 1 sec function

 push {r0,r1,r2,r3,r4,r5,r6,r7,r8}

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
  pop {r0,r1,r2,r3,r4,r5,r6,r7,r8}
 	bx lr
