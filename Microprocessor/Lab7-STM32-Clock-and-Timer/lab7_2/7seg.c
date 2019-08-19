#include "stm32l476xx.h"
#include "system_stm32l4xx.h"
#include "core_cmSimd.h"
#include "core_cmInstr.h"
#include "core_cmFunc.h"
#include "core_cm4.h"
#include "cmsis_gcc.h"
#include <stdio.h>
#include <stdlib.h>
#define SET_REG(REG,SELECT,VAL){( (REG) = ((REG)&(~(SELECT))) | (VAL) );};
#define TIME_SEC 12.70
#define TIM_COUNTERMODE_UP 0x0
#define TIM_ARR_VAL 40000
#define MSI_DEFAULT_FREQ 4000000
#define ZERO 0x7E
#define ONE 0x30
#define TWO 0x6D
#define TREE 0x79
#define FOUR 0x33
#define FIVE 0x5B
#define SIX 0x5F
#define SEVEN 0x70
#define EIGHT 0x7F
#define NINE 0x7B

#define ZERO_D 0xFE
#define ONE_D 0xB0
#define TWO_D 0xED
#define TREE_D 0xF9
#define FOUR_D 0xB3
#define FIVE_D 0xDB
#define SIX_D 0xDF
#define SEVEN_D 0xF0
#define EIGHT_D 0xFF
#define NINE_D 0xFB

int arr[2][10]={{ZERO,ONE,TWO,TREE,FOUR,FIVE,SIX,SEVEN,EIGHT,NINE},{ZERO_D,ONE_D,TWO_D,TREE_D,FOUR_D,FIVE_D,SIX_D,SEVEN_D,EIGHT_D,NINE_D}};


extern void max7219_init();
extern void max7219Send();

void display(int num){
	int addr,data;
	int deter=-1,num_temp;
	num_temp=num;
	while(1){
		num_temp=(num_temp/10);
		deter++;
		if(num_temp==0)break;
	}

	if(deter==0){ //0.01
		max7219Send(11,2);
		max7219Send(1,arr[0][num]);
		//add 0.0
		max7219Send(2,ZERO);
		max7219Send(3,ZERO_D);
	}
	else if(deter==1){ //0.12
		max7219Send(11,2);
		max7219Send(1,arr[0][num%10]);
		max7219Send(2,arr[0][num/10]);
		//add 0.0
		max7219Send(3,ZERO_D);
	}
	else{ //56.23
		max7219Send(11,deter);
		int temp=num;
		deter++;
		for(int i=1;i<=deter;i++){
			addr=i;
			data=temp%10;
			temp=temp/10;
			if(i==3){
				max7219Send(addr,arr[1][data]);
				continue;
			}
			max7219Send(addr,arr[0][data]);
		}
	}
}

void GPIO_init() {
	// SET gpio OUTPUT //
	RCC->AHB2ENR = RCC->AHB2ENR|0x2;//B
	//Set PB3.4.5 as output mode
	GPIOB->MODER = GPIOB->MODER&0xFFFFF03F;
	GPIOB->MODER |= 0xFFFFF57F;
}

void Timer_init()
{
//TODO: Initialize timer
	RCC->APB1ENR1 |= RCC_APB1ENR1_TIM2EN;
	SET_REG(TIM2->CR1, TIM_CR1_DIR , TIM_COUNTERMODE_UP);//up counter
	TIM2->ARR = (uint32_t)39999;//Reload value
	TIM2->PSC = (uint32_t)39999;//Prescalser
	TIM2->EGR = TIM_EGR_UG;//Reinitialize the counter
}

void Timer_start(){
//TODO: start timer and show the time on the 7-SEG LED.
	TIM2->CR1 |= TIM_CR1_CEN;//start timer
}

int main(){
	GPIO_init();
	max7219_init();
	Timer_init();
	Timer_start();
	int pre_val = TIME_SEC*100;
	while(1){
			//TODO: Polling the timer count and do lab requirements
			int timerValue = TIM2->CNT;//polling the counter value
			if(pre_val < timerValue){//check if times up
				TIM2->CR1 &= ~TIM_CR1_CEN;
				break;
			}
			display(timerValue);//display the time on the 7-SEG LED
		}
}


