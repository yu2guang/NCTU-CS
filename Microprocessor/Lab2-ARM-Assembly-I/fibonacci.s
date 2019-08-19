	.syntax unified
	.cpu cortex-m4
	.thumb
.text
	.global main
	.equ N,46
 fib:
 	//TODO
 	movs r1,#0
 	movs r4,#1 //r4: result
 loop:
 	mov r2,r4 //r2: 2nd
 	adds r4,r1,r4 //r4: 3rd
 	bvs overflow
 	mov r1,r2 //r1: 1st
 	subs r0,#1
 	cmp r0,#1
 	bne loop

 	bl L
 out:
 	movs r4,#-1
 	bl L
 overflow:
 	movs r4,#-2
 	bl L
 main:
 	movs r0,#N
 	cmp r0,#100
 	bgt out // >
 	cmp r0,#1
 	blt out // <
 	bl fib
	
L: b L
