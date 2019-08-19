#include <stdio.h>
#include <stdlib.h>
#include "stm32l476xx.h"
#include "system_stm32l4xx.h"
#include "core_cmSimd.h"
#include "core_cmInstr.h"
#include "core_cmFunc.h"
#include "core_cm4.h"
#include "cmsis_gcc.h"
#include<string.h>

#define SET_REG(REG,SELECT,VAL){((REG)=((REG)&(~(SELECT)))|(VAL));};
char arr[100]= "pinyugly";

void GPIO_Init(void) {
	RCC->AHB2ENR |= (RCC_AHB2ENR_GPIOAEN | RCC_AHB2ENR_GPIOBEN | RCC_AHB2ENR_GPIOCEN);

	// AF MODE: PC10, 11
	SET_REG(GPIOC->MODER,0x00F00000,0x00A00000);
	// AFRH: AFSEL10, 11; AF7
	SET_REG(GPIOC->AFR[1],0x0000FF00,0x00007700);

	/* BUTTON */
	SET_REG(GPIOC->MODER,0x0C000000,0);
}

void USART3_Init(void) {
	// RCC_APB1SMENR1_USART3SMEN

	/* Enable clock for USART3 */
	RCC->APB1ENR1 |= RCC_APB1ENR1_USART3EN;
    
	/* CR1 */
    // USART disable
	SET_REG(USART3->CR1,USART_CR1_UE,0);
    // M bit 00->8bit, word length
	SET_REG(USART3->CR1,USART_CR1_M,0);
    // oversampling=16
	SET_REG(USART3->CR1,USART_CR1_OVER8,0);
    //Baud=fck/USARTDIV
    //USARTDIV=417    //fck=4M    //Baud=9600
	SET_REG(USART3->BRR,USART_BRR_DIV_MANTISSA,4000000L/576000L);
    
	/* CR2 */
    // 1-bit stop
	SET_REG(USART3->CR2, USART_CR2_STOP, 0x0);

	// Enable UART
	USART3->CR1 |= (USART_CR1_UE);
}


void UART_Transmit(char one){

    SET_REG(USART3->CR1,USART_CR1_TE,USART_CR1_TE);
    int TE_bit;
    while(1){
        TE_bit = (USART3->CR1 >> 3) & 1;
        if(TE_bit==1) break;
    }
    
    USART3->TDR = one;

	int TC_bit;
	while (1){
		TC_bit = (USART3->ISR >> 6) & 1;
		if(TC_bit==1) break;
	}
}

int main(void){
	int is_button=1;
	GPIO_Init();
	USART3_Init();

	while(1){
		if((GPIOC->IDR)>>13==0)break;
	}
    
	for(int i=0;i<strlen(arr);i++){
		UART_Transmit(arr[i]);
	}

	return 0;
}
