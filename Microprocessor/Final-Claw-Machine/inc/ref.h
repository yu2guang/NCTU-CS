#ifndef REF_H_
#define REF_H_

typedef __INT32_TYPE__ int32_t ;
typedef __UINT32_TYPE__ uint32_t ;
typedef __INT8_TYPE__ int8_t ;
typedef __UINT8_TYPE__ uint8_t ;
#define     __IO    volatile
#define     __IM     volatile const      /*! Defines 'read only' structure member permissions */
#define     __OM     volatile            /*! Defines 'write only' structure member permissions */
#define     __IOM    volatile            /*! Defines 'read / write' structure member permissions */

/*********************RCC Define*********************/


static inline void delay_us(int n){
	asm("push {r0}\r\n"
			"mov r0, r0\r\n"
			"LOOP_US:\r\n"
			"nop\r\n"
			"subs r0, #1\r\n"
			"BGT LOOP_US\r\n"
			"POP {r0}\r\n"
			:: "r" (n));
}

static inline void delay_ms(int n){
	asm("push {r0}\r\n"
			"mov r0, %0\r\n"
			"LOOP:\r\n"
			"subs r0, #1\r\n"
			"BGT LOOP\r\n"
			"POP {r0}\r\n"
			:: "r" (n*1333));
}



#endif /* REF_H_ */
