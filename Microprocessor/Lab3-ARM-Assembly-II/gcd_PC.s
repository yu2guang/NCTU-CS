	.syntax unified
	.cpu cortex-m4
	.thumb
.data
	result: .word  0
	max_size:  .word  0
.text
	.global main
	m: .word  0x5E // 94
	n: .word  0x60 // 96
main:
	ldr r2, =m
	ldr r0, [r2]
	ldr r2, =n
	ldr r1, [r2]
	mov r7,sp //calculate stack size
	mov r8,r7 //r8:temp sp
	mov r5,PC
	add r5,#11
	push {r5}
	push {r0,r1}
	bl GCD

	subs r7,r7,r8
	lsr r7,r7,2
	ldr r1,=max_size
	str r7,[r1]//r1:max_size adr r7:max_size
	ldr r2,=result
	str r6,[r2]//r2:result adr r6:result
L:
	B L

GCD:
	//TODO: Implement your GCD function
	// stack contains m, n, lr
	cmp sp,r8
	IT lt
	movlt r8,sp
	pop {r0,r1} // r0: a, r1: b

	// if(a == 0) return b;
case1:
	cmp r0, #0
	bne case2
	pop {r2} // pop lr
	mov r6,r1
	mov PC,r2

	// if(b == 0) return a;
case2:
	cmp r1, #0
	bne case3
	pop {r2} // pop lr
	mov r6,r0
	mov PC,r2

case3:
	and r2,r0,#1 // r2: a%2
	and r3,r1,#1 // r3: b%2
	orr r4,r2,r3 // r4: if=0, a%2==0 && b%2==0

	// if(a%2==0 && b%2==0) return 2*gcd(a>>1, b>>1);
	cmp r4, #0
	bne case4
	lsr r0,r0,1
	lsr r1,r1,1
	mov r5,PC
	add r5,#9
	push {r5}
	push {r0,r1}
	b GCD

	mov r1,#2
	mul r6,r1
	pop {r2} // pop lr
	mov PC,r2

	// else if(a%2==0) return gcd(a>>1, b);
case4:
	cmp r2,#0
	bne case5
	lsr r0,r0,1

    // else if(b%2==0) return gcd(a, b >> 1);
case5:
	cmp r3,#0
	bne case6
	lsr r1,r1,1

    // else return gcd(abs(a - b), Min(a, b))
case6:
	and r4,r2,r3 // r4: if=1, a%2==1 && b%2==1
	cmp r4,#1
	bne no_mul2
	cmp r0,r1
	IT gt
	subgt r0,r0,r1 // >
	IT lt
	sublt r1,r1,r0 // <
	IT eq
	subeq r0,r0,r1 // =

no_mul2:
	mov r5,PC
	add r5,#9
	push {r5}
	push {r0,r1}
	b GCD
	pop {r2} // pop lr
	mov PC,r2


/*
int Gcd(int a, int b)
{
    if(a == 0) return b;
    if(b == 0) return a;

    if(a % 2 == 0 && b % 2 == 0){
    	return 2 * gcd(a >> 1, b >> 1);
    }
    else if(a % 2 == 0){
    	return gcd(a >> 1, b);
    }
    else if(b % 2 == 0){
    	return gcd(a, b >> 1);
    }
    else{
    	return gcd(abs(a - b), Min(a, b));
    }
}
*/
