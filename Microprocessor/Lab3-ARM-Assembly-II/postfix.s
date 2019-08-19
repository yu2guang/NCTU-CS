	.syntax unified
	.cpu cortex-m4
	.thumb
.data
	user_stack: .zero 128
	expr_result: .word 0

.text
	.global main
	postfix_expr: .asciz "-100 10 20 + - 10 +"
	.align 2
	//minus: .asciz "-"//45
	//blank: .asciz " "//32
	//add: .asciz "+"//43
	.equ minus,0x2D
	.equ blank,0x20
	.equ add,0x2B
	.equ to_int,0x40030

main:
	ldr   r0, =user_stack
	add r0,r0,128
  	msr   msp, r0

	LDR R0, =postfix_expr
	movs r1,r0 // r1:string pointer, i
	ldrb r4,[r1] // r4:string pointer char, s[i]

spilt_str: // for loop

	// initialize
	movs r2,#0 // r2:negative
	movs r3,#0 // r3:push_a

point_char: // while loop
//is minus
	cmp r4,#minus // '-'
	bne non_negative

	// determine operater
	add r5,r1,#1 // r5:i+1
	ldrb r6,[r5]  // r6:s[i+1]

	cmp r6,#blank
	IT eq
	moveq r7,#1 // r7=1, if r6==' '
	cmp r6,#0
	IT eq
	moveq r8,#1 // r8=1, if r6=='0'
	orr r8,r7,r8 // r8=1, if r6==' 'or'0'
	cmp r8,#1
	ITTT ne
	movne r2,#1
	addne r1,r1,#1 // i++
	ldrbne r4,[r1]
	bne point_char // negative int

	// start minus
	pop {r5} // r5: n2
	pop {r6} // r6: n1
	sub r5,r6,r5
	push {r5}
	add r1,r1,#1
	ldrb r4,[r1]
	cmp r4,#blank
	ITT eq
	addeq r1,r1,#1
	ldrbeq r4,[r1]
	cmp r4,#0 // cmp s[i],0
	beq result
	b spilt_str

non_negative:	//is add
	cmp r4,#add // '+'
	bne non_add
	// start add
	pop {r5} // r5: n2
	pop {r6} // r6: n1
	add r5,r6,r5
	push {r5}
	add r1,r1,#1
	ldrb r4,[r1]
	cmp r4,#blank
	ITT eq
	addeq r1,r1,#1
	ldrbeq r4,[r1]
	cmp r4,#0 // cmp s[i],0
	beq result
	b spilt_str

non_add: //is digit
	// char to int
	bl atoi
	mov r6,#10
	mul r3,r6 // push_a *= 10
	add r3,r5,r3 // push_a += temp

	add r1,r1,#1 // i++
	ldrb r4,[r1]
	cmp r4,#0 // cmp s[i],0
	beq result
	cmp r4,#blank // ' '
	bne point_char

	cmp r2,#1
	mov r6,r3 // push_a -= (push_a*2);
	ITT eq
	subeq r3,r6
	subeq r3,r6
	push {r3}

	add r1,r1,#1 // i++
	ldrb r4,[r1]
	cmp r4,#0 // cmp s[i],0
	bne spilt_str

result:
	pop {r0}
	ldr r1,=expr_result
	str r0,[r1]

//TODO: Setup stack pointer to end of user_stack and calculate the	expression using PUSH, POP operators, and store the result into	expr_result


program_end:
	B program_end

atoi:
 //TODO: implement a ¡§convert string to integer¡¨ function
	sub r5,r4,#'0'
	BX LR
