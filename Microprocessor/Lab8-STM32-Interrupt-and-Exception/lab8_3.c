#include <stdio.h>
#include <stdlib.h>
#include "stm32l476xx.h"
#include "system_stm32l4xx.h"
#include "core_cmSimd.h"
#include "core_cmInstr.h"
#include "core_cmFunc.h"
#include "core_cm4.h"
#include "cmsis_gcc.h"

//extern void Delay();
#define SET_REG(REG,SELECT,VAL){((REG)=((REG)&(~(SELECT)))|(VAL));};
int keypad_value[4][4]={{1,2,3,10},{4,5,6,11},{7,8,9,12},{15,0,14,13}};
int keypad_value_now=0;
int button_pause = 0;


void Timer_init(){
	//TODO: Initialize timer
	//TIM2 enable
	RCC->APB1ENR1 |= RCC_APB1ENR1_TIM2EN;
	//prescaler
	TIM2->PSC = (uint32_t)99;
	//reload value
	TIM2->ARR = (uint32_t)99;
	TIM2->EGR = TIM_EGR_UG; //Reinitialize the counter. CNT takes the auto-reload value.
	TIM2->CR1 |= TIM_CR1_CEN; //start timer
}

void PWM_channel_init(){
	//TODO: Initialize timer PWM channel
	/*
		Set TIMX start=off
		Set TIMX enable
		Set TIMX prescaler , reload value, count_dir
		Set TIMX capture/compare enable
		Set TIMX capture/compare as output
		Set TIMX capture/compare as pwm mode
		Set TIMX capture/compare reg?嚙踝� count value ???
		Set TIMX re-initialize counter=on
		Set TIMX update interrupt enable
		Set TIMX start=on
	*/
	//SET_REG(TIM2->CR1,TIM_CR1_UDIS,TIM_CR1_UDIS);
	//set TIM2 start=off
	SET_REG(TIM2->CR1,TIM_CR1_CEN,0);
	//TIM2 enable
	RCC->APB1ENR1 |= RCC_APB1ENR1_TIM2EN;
	//count_dir; DIR: direction(up), CMS: center-aligned mode selection(edge)
	SET_REG(TIM2->CR1,TIM_CR1_DIR | TIM_CR1_CMS,0);
	/*
		capture/compare output, PWM mode
		disable CC3E=0 to set CC3S
		output -> CC3S=00
		no accelerate -> OC3FE=0
		preload enable -> OC3PE=1
		clear enable (affected by ETRF) -> OC3CE=0
		PWM mode1 -> OC3M=0110
	*/
	SET_REG(TIM2->CCER,TIM_CCER_CC3E,0);
	SET_REG(TIM2->CCMR2, TIM_CCMR2_CC3S|TIM_CCMR2_OC3FE|TIM_CCMR2_OC3PE|TIM_CCMR2_OC3M|TIM_CCMR2_OC3CE, TIM_CCMR2_OC3PE|(TIM_CCMR2_OC3M_1|TIM_CCMR2_OC3M_2));
	/*
		capture/compare enable
		TIM2 -> CC3E; output clear -> CC3NP=1
		active high -> CC3P=0
		OC3 active -> CC3E=1
	*/
	SET_REG(TIM2->CCER,TIM_CCER_CC3NP|TIM_CCER_CC3P|TIM_CCER_CC3E,TIM_CCER_CC3NP|TIM_CCER_CC3E);
	//capture/compare reg's count value
	//set CCR
	SET_REG(TIM2->CCR3,TIM2->CCR3,50);
	//reinitialize the counter
	TIM2->EGR = TIM_EGR_UG;
	//update interrupt enable <- why?
	SET_REG(TIM2->DIER,TIM_DIER_UIE,TIM_DIER_UIE);
	//set TIM2 start=on
	SET_REG(TIM2->CR1,TIM_CR1_CEN,TIM_CR1_CEN);
}


void EXTI_config(){
	// SYSCFGEN
	SET_REG(RCC->APB2ENR,1,1);

	// set GPIO PC13 as EXTIx source
	SET_REG(SYSCFG->EXTICR[3],0xF0,SYSCFG_EXTICR4_EXTI13_PC);

	// select rising edge
	SET_REG(EXTI->RTSR1,EXTI_RTSR1_RT13,EXTI_RTSR1_RT13);
	SET_REG(EXTI->FTSR1,EXTI_FTSR1_FT13,0);

	// enable this interrupt
	SET_REG(EXTI->IMR1,EXTI_IMR1_IM13,EXTI_IMR1_IM13);

	// pending flag
	SET_REG(EXTI->PR1,EXTI_PR1_PIF13|EXTI_PR1_PIF10|EXTI_PR1_PIF11|EXTI_PR1_PIF12|EXTI_PR1_PIF14|EXTI_PR1_PIF15,EXTI_PR1_PIF13|EXTI_PR1_PIF10|EXTI_PR1_PIF11|EXTI_PR1_PIF12|EXTI_PR1_PIF14|EXTI_PR1_PIF15);
}

void NVIC_config(){
	// enable IRQ
	//NVIC_EnableIRQ(EXTI15_10_IRQn);
	SET_REG(NVIC->ISER[1],(1<<8),(1<<8)); // ISER[1]: 32-bit + 8
	//SET_REG(NVIC->IP[39],0b10000,0b10000); // 40/4(bit) = 10...0(byte); register: 10, offset: 0->0
	NVIC_SetPriority(40, 1); // button
}

void EXTI15_10_IRQHandler(void){
	// unable systik
	SET_REG(SysTick->CTRL,SysTick_CTRL_ENABLE_Msk,0);
	// interrupt turn off
	SET_REG(EXTI->PR1,EXTI_PR1_PIF13|EXTI_PR1_PIF10|EXTI_PR1_PIF11|EXTI_PR1_PIF12|EXTI_PR1_PIF14|EXTI_PR1_PIF15,EXTI_PR1_PIF13|EXTI_PR1_PIF10|EXTI_PR1_PIF11|EXTI_PR1_PIF12|EXTI_PR1_PIF14|EXTI_PR1_PIF15);
	SET_REG(RCC->APB1ENR1,RCC_APB1ENR1_TIM2EN,0);

	button_pause = 1;

	//sdfjkljklsdfjklsdfksdfjlfjklsdjklsdffjklsdklsdfjjkldfssfkldjsdfjkljsdfkllfjksdsdfjklfjklsdjklsdf
}

void GPIO_init_AF(){
	//TODO: Initial GPIO pin as alternate function for buzzer.
	/*
		pin10 -> AFRH(AFR[1]) -> AFSEL10
		PB10,TIM2_CH3 -> AF1
	*/
	GPIOB->MODER &= 0xFFCFFFFF;
	GPIOB->MODER |= 0x00200000;
	GPIOB->AFR[1] |= GPIO_AFRH_AFSEL10_0; //zooah
}

void SysTick_Handler(void){

	keypad_value_now--;

	if(keypad_value_now>0) return;

	while(1){
		if(button_pause==1){
			button_pause = 0;
			SET_REG(RCC->APB1ENR1,RCC_APB1ENR1_TIM2EN,0);
			return;
		}
		else{
			RCC->APB1ENR1 |= RCC_APB1ENR1_TIM2EN;
		}
	}
}

void Init_SysTick(){
	// enable SysTick
	SET_REG(SysTick->CTRL,SysTick_CTRL_ENABLE_Msk,SysTick_CTRL_ENABLE_Msk);
	// set reload register
	SET_REG(SysTick->CTRL,SysTick_CTRL_TICKINT_Msk,SysTick_CTRL_TICKINT_Msk); // exception enable
	SET_REG(SysTick->CTRL,SysTick_CTRL_COUNTFLAG_Msk,SysTick_CTRL_COUNTFLAG_Msk); // VAL=0, reload
	// Load the SysTick Counter Value
	SysTick->LOAD = (uint32_t)4000000-1;
	// CLKSOURCE processor clock
	SET_REG(SysTick->CTRL,SysTick_CTRL_CLKSOURCE_Msk,SysTick_CTRL_CLKSOURCE_Msk);
	// systik priority
	//SET_REG(SCB->SHP[11],0b11111111,0b11111111);
	NVIC_SetPriority(SysTick_IRQn, 2);
}

void init_GPIO(){
	// SET keypad gpio OUTPUT //
	RCC->AHB2ENR = RCC->AHB2ENR|0x7;
	//Set PA5,6,7,8 as output mode
	GPIOA->MODER= GPIOA->MODER&0xFFFD57FF;
	//set PA5,6,7,8 is Pull-up output
	GPIOA->PUPDR=GPIOA->PUPDR|0x15400;
	//Set PA5,6,7,8 as medium speed mode
	GPIOA->OSPEEDR=GPIOA->OSPEEDR|0x15400;
	//Set PA5,6,7,8 as high
	GPIOA->ODR=GPIOA->ODR|1111<<5;

	// SET LED gpio OUTPUT, PB10 //
	GPIOB->MODER= GPIOB->MODER&0xFFCFFFFF;
	GPIOB->MODER |= 0x20000;

	// SET keypad gpio INPUT //
	//Set PC5,6,7,8,13 as INPUT mode
	GPIOC->MODER=GPIOC->MODER&0xF3FC03FF;
	//set PC5,6,7,8 is Pull-down input
	GPIOC->PUPDR=GPIOC->PUPDR|0x2A800;
	//Set PC5,6,7,8 as medium speed mode
	GPIOC->OSPEEDR=GPIOC->OSPEEDR|0x15400;
}

void keypad_scan(){
	unsigned int flag_keypad,flag_debounce;
		while(1){
			flag_keypad=GPIOC->IDR&0b1111<<5;
			if(flag_keypad!=0){
				int k=45000;
				while(k!=0){
					flag_debounce=GPIOC->IDR&0b1111<<5;
					k--;
				}
				if(flag_debounce!=0){
					int position_c;
					for(int i=0;i<4;i++){ //scan keypad from first column
						position_c=i+5;
						//set PA5,6,7,8(column) low and set pin high from PA5
						GPIOA->ODR=(GPIOA->ODR&0xFFFFFE1F)|1<<position_c;
						int position_r;
						unsigned int flag_keypad_r;
						for(int j=0;j<4;j++){ //read input from first row
							position_r=j+5;
							flag_keypad_r=GPIOC->IDR&(1<<position_r);
							if(flag_keypad_r!=0){
								keypad_value_now = keypad_value[j][i];
								if(keypad_value[j][i]!=0){
									Init_SysTick();
								}
							}
						}
					}
				}
			}

			GPIOA->ODR=GPIOA->ODR|0b1111<<5; //set PA5,6,7,8(column) high
		}
}

int main(){
	NVIC_config();
	init_GPIO();
	GPIO_init_AF();
	Timer_init();
	PWM_channel_init();

	SET_REG(RCC->APB1ENR1,RCC_APB1ENR1_TIM2EN,0);
	EXTI_config();

	keypad_scan();
}
