	.syntax unified
	.cpu cortex-m4
	.thumb
.data
	result: .byte 0
.text
	.global main
	.equ X, 0x55AA
	.equ Y, 0xAA55
hamm:
	//TODO
	eor r3,r0,r1 // r0 xor r1
	movs r4,#0 // r4:count
count:
	ands r5,r3,#1 // if r3 rightmost is 1
	cmp r5,#0
	beq zero
	adds r4,r4,#1 // count++
zero:
	lsr r3,r3,#1 // r3 >> 1
	cmp r3,#0
	bne count


	bx lr
main:
	movw R0, #X
	movw R1, #Y
	ldr R2, =result
	bl hamm
	str r4, [r2]
L: b L
