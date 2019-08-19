#include <stdio.h>
#include <stdlib.h>
#include "stm32l476xx.h"
#include "system_stm32l4xx.h"
#include "core_cmSimd.h"
#include "core_cmInstr.h"
#include "core_cmFunc.h"
#include "core_cm4.h"
#include "cmsis_gcc.h"

#define SET_REG(REG,SELECT,VAL) {( (REG) =( (REG) & (~(SELECT)) ) | (VAL) );};
#define X0 0b100000
#define X1 0b1000000
#define X2 0b10000000
#define X3 0b100000000
#define Y0 0b100000
#define Y1 0b1000000
#define Y2 0b10000000
#define Y3 0b100000000

//extern void GPIO_init();

unsigned int x_pin[4] = {X0, X1, X2, X3};
unsigned int y_pin[4] = {Y0, Y1, Y2, Y3};
int keypad_value[4][4]={{152,135,120,0},{114,101,90,0},{80,75,0,0},{1,0,2,0}};

void keypad_init() {

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

	GPIOB->MODER= GPIOB->MODER&0xFFCFFFFF;
	GPIOB->MODER |= 0x100000;

	// SET keypad gpio INPUT //
	//Set PC5,6,7,8 as INPUT mode
	GPIOC->MODER=GPIOC->MODER&0xFFFC03FF;
	//set PC5,6,7,8 is Pull-down input
	GPIOC->PUPDR=GPIOC->PUPDR|0x2A800;
	//Set PC5,6,7,8 as medium speed mode
	GPIOC->OSPEEDR=GPIOC->OSPEEDR|0x15400;

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
		Set TIMX capture/compare reg?™s count value ???
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
	//capture/compare reg?™s count value
	//set CCR // duty cycle 50%
	SET_REG(TIM2->CCR3,TIM2->CCR3,50);
	//reinitialize the counter
	TIM2->EGR = TIM_EGR_UG;
	//update interrupt enable <- why?
	SET_REG(TIM2->DIER,TIM_DIER_UIE,TIM_DIER_UIE);
	//set TIM2 start=on
	SET_REG(TIM2->CR1,TIM_CR1_CEN,TIM_CR1_CEN);
}

void PWM_send(float which_button){
	//TODO: Use PWM to send the corresponding frequency square wave to buzzer.
	// PSC = 4Mhz / (ARR*f)
	TIM2->PSC = (uint32_t)(4000000/(100*which_button));
}

void keypad_scan(){
	unsigned int flag_keypad,flag_debounce;
	int press_stat=0,befoer_val=-1,press_val=0;
	int CCR_value=50; //10 <= CCR_value <= 90

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
							if(keypad_value[j][i]==1){ //-5%
								if(CCR_value>10){
									CCR_value -= 5;
									SET_REG(TIM2->CCR3,TIM2->CCR3,CCR_value);
								}
							}
							else if(keypad_value[j][i]==2){ //+5%
								if(CCR_value<90){
									CCR_value += 5;
									SET_REG(TIM2->CCR3,TIM2->CCR3,CCR_value);
								}
							}
							if(keypad_value[j][i]!=0){
								press_stat = 1;
								if(befoer_val != -1)
									TIM2->CR1 |= TIM_CR1_CEN;
								if(befoer_val != keypad_value[j][i]){
									press_val = keypad_value[j][i];
									TIM2->PSC = keypad_value[j][i];
							    }
							}
						}
					}
				}
			}
		}
		if(press_stat == 1)
			befoer_val = press_val;
		else{
			TIM2->CR1 &= ~TIM_CR1_CEN;
			befoer_val = -1;
		}
		press_stat = 0;

		GPIOA->ODR=GPIOA->ODR|0b1111<<5; //set PA5,6,7,8(column) high
	}
}

int main(){

	keypad_init();
	GPIO_init_AF();
	Timer_init();
	PWM_channel_init();

	while(1){
		keypad_scan();
	}
}
