#include <stdio.h>
#include <stdlib.h>
#include <system_stm32l4xx.h>
#include <stm32l476xx.h>
#include <core_cmSimd.h>
#include <core_cmInstr.h>
#include <core_cmFunc.h>
#include <core_cm4.h>
#include <cmsis_gcc.h>

extern void Delay();
#define SET_REG(REG,SELECT,VAL){((REG)=((REG)&(~(SELECT)))|(VAL));};

int user_press_button(){
	int pressed = 1;
	pressed = !((GPIOC->IDR>>13) && 1);
	return pressed;
}

void SystemClock_Config(){
	//TODO: Change the SYSCLK source and set the corresponding Prescaler value.
	RCC->CR |= RCC_CR_HSION;// turn on HSI16 oscillator
	//RCC->CR |= RCC_CR_MSION;// turn on HSI16 oscillator
	while((RCC->CR & RCC_CR_HSIRDY) == 0);//check HSI16 ready
	//while((RCC->CR & RCC_CR_MSIRDY) == 0);//check HSI16 ready
	SET_REG(RCC->CFGR, RCC_CFGR_HPRE, 5<<4);//SYSCLK divide by 16. SYSCLK = 16MHz/16 = 1Mhz     //HPRE:AHB prescaler 11
	if((RCC->CR & RCC_CR_HSIRDY) == 0)	return;
	//if((RCC->CR & RCC_CR_MSIRDY) == 0)	return;

	// Use HSI16 as system clock
	//APB1 prescaler not divide
	//APB2 prescaler not divide
}

void GPIO_init() {
	// SET gpio OUTPUT //
	RCC->AHB2ENR = RCC->AHB2ENR|0x5;//A&C
	//Set PA5 as output mode
	GPIOA->MODER= GPIOA->MODER&0xFFFFF7FF;


	// SET user button gpio INPUT //
	//Set PC 13 INPUT mode
	GPIOC->MODER=GPIOC->MODER&0xF3FFFFFF;
}

int main(){

	 SystemClock_Config();
	 GPIO_init();
int i;
	 for( i=0;;){
		 if(i==0){
			 RCC->CFGR &= ~RCC_CFGR_SW;
			 RCC->CFGR |= RCC_CFGR_SW_MSI;
			 while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_MSI);

			 RCC->CR &= 0xFEFFFFFF;
			 while((RCC->CR && RCC_CR_PLLRDY) == 0);

			 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLN, 8<<8);
			 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLM, 3<<4);//1
			 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLR, 3<<25);

			 SET_REG(RCC->PLLCFGR,RCC_PLLCFGR_PLLSRC,RCC_PLLCFGR_PLLSRC_MSI);
			 RCC->CR |= RCC_CR_PLLON;
			 SET_REG(RCC->PLLCFGR,RCC_PLLCFGR_PLLPEN,RCC_PLLCFGR_PLLPEN);
			 SET_REG(RCC->PLLCFGR,RCC_PLLCFGR_PLLQEN,RCC_PLLCFGR_PLLQEN);
			 SET_REG(RCC->PLLCFGR,RCC_PLLCFGR_PLLREN,RCC_PLLCFGR_PLLREN);

			 RCC->CFGR &= ~RCC_CFGR_SW;
			 RCC->CFGR |= RCC_CFGR_SW_PLL;
			 while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL);
			 i++;
		 }

		 if (user_press_button())
		 {
			 //TODO: Update system clock rate
			 RCC->CFGR &= ~RCC_CFGR_SW;
			 RCC->CFGR |= RCC_CFGR_SW_MSI;
			 while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_MSI);

			 RCC->CR &= 0xFEFFFFFF;
			 while((RCC->CR && RCC_CR_PLLRDY) == 0);
			 if(i%5==1){ // -> 6Mhz
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLN, 24<<8);
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLM, 3<<4);//PLLM=4
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLR, 1<<25);//4
				 i++;
			 }
			 else if(i%5==2){ // -> 10Mhz
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLN, 20<<8);
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLM, 0<<4);//1
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLR, 3<<25);//8
				 i++;
			 }
			 else if(i%5==3){ // -> 16Mhz
			 	SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLN, 32<<8);
			 	SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLM, 3<<4);//4
			 	SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLR, 0<<25);//2
				i++;
			 }
			 else if(i%5==4){ // -> 40Mhz
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLN, 40<<8);
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLM, 1<<4);//2
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLR, 0<<25);//2
				 i++;
			 }
			 else{ // -> 1Mhz
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLN, 8<<8);
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLM, 3<<4);
				 SET_REG(RCC->PLLCFGR, RCC_PLLCFGR_PLLR, 3<<25);
				 i++;
			 }
			 //RCC_PLLCFGR_PLLSRC_MSI;
			 SET_REG(RCC->PLLCFGR,RCC_PLLCFGR_PLLSRC,RCC_PLLCFGR_PLLSRC_MSI);
			 RCC->CR |= RCC_CR_PLLON;
			 SET_REG(RCC->PLLCFGR,RCC_PLLCFGR_PLLPEN,RCC_PLLCFGR_PLLPEN);
			 SET_REG(RCC->PLLCFGR,RCC_PLLCFGR_PLLQEN,RCC_PLLCFGR_PLLQEN);
			 SET_REG(RCC->PLLCFGR,RCC_PLLCFGR_PLLREN,RCC_PLLCFGR_PLLREN);

			 RCC->CFGR &= ~RCC_CFGR_SW;
			 RCC->CFGR |= RCC_CFGR_SW_PLL;
			 while ((RCC->CFGR & RCC_CFGR_SWS) != RCC_CFGR_SWS_PLL);

		 }
		 GPIOA->BSRR = (1<<5);
		 Delay();
		 GPIOA->BRR = (1<<5);
		 Delay();
	 }
 }
