	.syntax unified
	.cpu cortex-m4
	.thumb
 .data
	user_stack: .zero 400
 .text
	.global main
	.equ n, 16

 stair:
 	mov r0,lr
 	push {r0}

 case1: // r3 = 0
 	cmp r3,#0
 	bne case2
 	adds r1,#1
 	pop {r0} // pop lr
	mov PC,r0

 case2: // r3 < 0
 	cmp r3,#0
 	bgt case3
 	pop {r0} // pop lr
	mov PC,r0

 case3:
 	push {r3}
 	sub r3,#1
 	bl stair

 	pop {r3}
 	sub r3,#2
    cmp r3,#0
 	bge no
 	pop {r0} // pop lr
	mov PC,r0
	no:
 	push {r3}
 	bl stair
 	pop {r3}
 	pop {r0} // pop lr
	mov PC,r0

 main:
 	ldr r0,=user_stack
	add r0,r0,400
  	msr msp,r0

	mov r1,#0 // r1: result
	mov r3,#n

	cmp r3,#0
	bge positive // >=
	mov r1,#-1
	b L
 positive:
	bl stair
 L: B L
