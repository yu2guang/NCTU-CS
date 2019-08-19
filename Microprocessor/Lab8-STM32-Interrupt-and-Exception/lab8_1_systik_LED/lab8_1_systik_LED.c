#include <stdio.h>
#include <stdlib.h>
#include "stm32l476xx.h"
#include "system_stm32l4xx.h"
#include "core_cmSimd.h"
#include "core_cmInstr.h"
#include "core_cmFunc.h"
#include "core_cm4.h"
#include "cmsis_gcc.h"

#define SET_REG(REG,SELECT,VAL){((REG)=((REG)&(~(SELECT)))|(VAL));};

void SysTick_Handler(void){
	//LED control
	GPIOA->ODR ^= (0b1<<5);
}

void SystemClock_Config_source(void){
	RCC->CR |= RCC_CR_HSION;// turn on HSI16 oscillator

	RCC->CFGR &= ~RCC_CFGR_SW;
	RCC->CFGR |= RCC_CFGR_SW_HSI;
	while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_HSI);

	while((RCC->CR & RCC_CR_HSIRDY) == 0);//check HSI16 ready
	SET_REG(RCC->CFGR, RCC_CFGR_HPRE, 9<<4);//SYSCLK divide by 16. SYSCLK = 16MHz/4 = 4Mhz     //HPRE:AHB prescaler 9
	if((RCC->CR & RCC_CR_HSIRDY) == 0)	return;
}

void Init_SysTick(){
	// enable SysTick
	SET_REG(SysTick->CTRL,SysTick_CTRL_ENABLE_Msk,SysTick_CTRL_ENABLE_Msk);
	// set reload register
	SET_REG(SysTick->CTRL,SysTick_CTRL_TICKINT_Msk,SysTick_CTRL_TICKINT_Msk); // exception enable
	SET_REG(SysTick->CTRL,SysTick_CTRL_COUNTFLAG_Msk,SysTick_CTRL_COUNTFLAG_Msk); // VAL=0, reload
	//-------// Load the SysTick Counter Value
	SysTick->LOAD = (uint32_t)4000000*3-1; //think 1 think
	// CLKSOURCE processor clock
	SET_REG(SysTick->CTRL,SysTick_CTRL_CLKSOURCE_Msk,SysTick_CTRL_CLKSOURCE_Msk);
}

void init_GPIO(){
	RCC->AHB2ENR = RCC->AHB2ENR|0x1; //A
	//Set PA5 as output mode
	GPIOA->MODER= GPIOA->MODER&0xFFFFF7FF;
}

int main(){
	SystemClock_Config_source();
	Init_SysTick();
	init_GPIO();
}
