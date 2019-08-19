	.syntax unified
	.cpu cortex-m4
	.thumb
 .data
    operand_stack_bottom: .zero 16
    operand_stack_pointer: .float 3.14, 2.7, 9.8, 1.1
    operand_stack_top:
    operators: .asciz "+*+"
 .text
 	.global main
 /*
 	don't change
 	r3: # of operand
 */
 main:
 	// change sp
	ldr r0, =operand_stack_pointer
  	msr msp, r0

	// enable FPU
	ldr.w r0,=0xE000ED88
	ldr r1,[r0]
	orr r1,r1,#(0xF<<20)
	str r1,[r0]
	dsb
	isb

	ldr r0, =operators
	start:
		ldrb r1,[r0]
		cmp r1,#0
		beq L

		vpop {s0,s1}
		/* float num */
		cmp r1,#'+'
		IT eq
		vaddeq.f32 s0,s1
		cmp r1,#'*'
		IT eq
		vmuleq.f32 s0,s1
		vpush {s0}
		add r0,#1
	b start

 L: b L





