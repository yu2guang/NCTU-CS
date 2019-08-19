	.syntax unified
	.cpu cortex-m4
	.thumb
.data
	arr1: .byte 0x19, 0x34, 0x14, 0x32, 0x52, 0x23, 0x61, 0x29
	arr2: .byte 0x18, 0x17, 0x33, 0x16, 0xFA, 0x20, 0x55, 0xAC
	.equ number, 0x8
.text
	.global main

do_sort:
	//TODO
	movs r1, #number // r1: i=8

loop_out:
	movs r2, #0 // r2: j
	movs r3, #1 // r3: j+1
loop_in:
	ldrb r4,[r0,r2] // r4: arr[j] value
	ldrb r5,[r0,r3] // r5: arr[j+1] value
	cmp r4,r5
	ble unchange // arr[j]<=arr[j+1]
	strb r4,[r0,r3] // arr[j+1] = arr[j]
	strb r5,[r0,r2] // arr[j] = arr[j+1]

unchange:
	add r2, #1 // j++
	add r3, #1 // (j+1)++
	cmp r3, r1
	blt loop_in // j<i

	sub r1, #1
	cmp r1, #0
	bne loop_out

	bx lr
main:
	ldr r0, =arr1  // r0: start of arr
	bl do_sort
	ldr r0, =arr2
	bl do_sort
L: b L
