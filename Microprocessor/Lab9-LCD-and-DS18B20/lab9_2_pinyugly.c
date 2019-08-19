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
#define LCD_RSPin 0x2000 //PB13
#define LCD_RWPin 0x4000 //PB14
#define LCD_ENPin 0x8000 //PB15
int LCD_dataPin[8] = {0b1,0b10,0b100,0b1000,0b10000,0b100000,0b1000000,0b10000000}; //D0~D7: PB0~PB7
int cur_pos;
char str_arr[16] = "pinyugly";
int total_num;
int mode = 0; // 0:group #; 1:string
int cur_digit;

extern void Delay();

/**********************************
 *
 * LCD VO		PB8
 * LCD RS		PB13	output
 * LCD RW		PB14	output
 * LCD E		PB15	output
 * LCD D0		PB0		output
 * LCD D1		PB1		output
 * LCD D2		PB2		output
 * LCD D3		PB3		output
 * LCD D4		PB4		output
 * LCD D5		PB5		output
 * LCD D6		PB6		output
 * LCD D7		PB7		output
 *
 **********************************/


int WriteToLCD(int input, int isCmd){
	//TODO: Write command to LCD or let LCD display character fetched from memory.
	SET_REG(GPIOB->ODR,0b111000011111111,0);//clear ODR

	if(isCmd==0){ //...
		//TM_GPIO_SetPinLow(GPIOC,(uint32_t)LCD_RSPin);
		SET_REG(GPIOB->ODR,LCD_RSPin,0);
	}else if(isCmd==1){ //...
		//TM_GPIO_SetPinHigh(GPIOC,(uint32_t)LCD_RSPin);
		SET_REG(GPIOB->ODR,LCD_RSPin,LCD_RSPin);
	}

	//TM_GPIO_SetPinLow(GPIOC, LCD_RWPin);
	SET_REG(GPIOB->ODR,LCD_RWPin,0);

	int input_bit;
	for (int index = 0 ; index < 8 ; index++){
		input_bit = (input>>index) & 1;
		if(input_bit==1){ //...
			//TM_GPIO_SetPinHigh(GPIOA,(uint32_t)LCD_dataPin[index]);
			SET_REG(GPIOB->ODR,LCD_dataPin[index],LCD_dataPin[index]);
		}
		else if(input_bit==0){ //...
			//TM_GPIO_SetPinLow(GPIOA,(uint32_t)LCD_dataPin[index]);
			SET_REG(GPIOB->ODR,LCD_dataPin[index],0);
		}
		//...
	}

	//TM_GPIO_SetPinHigh(GPIOC,(uint32_t)LCD_ENPin);
	SET_REG(GPIOB->ODR,LCD_ENPin,LCD_ENPin);
	Delay();
	//TM_GPIO_SetPinLow(GPIOC,(uint32_t)LCD_ENPin);
	SET_REG(GPIOB->ODR,LCD_ENPin,0);
	Delay();

	return 0;
}

void init_LCD() {
	// LCD Register
	WriteToLCD(0x38, 0); // Function Setting: two row
	WriteToLCD(0x06, 0); // Entering Mode
	WriteToLCD(0x0E, 0); // Display on
	WriteToLCD(0x01, 0); // Clear Screen
	WriteToLCD(0x80, 0); // Move to top left

	cur_pos = 0x80;
}

void SysTick_Handler(void){

	int chan_mode = (GPIOC->IDR >> 13) & 1;

	if(chan_mode==0){
		mode ^= 1;
		WriteToLCD(0x01, 0); // Clear Screen
		WriteToLCD(0x80, 0); // Move to top left
		cur_digit = 0;
	}

	if(mode==0){
		WriteToLCD(0x01, 0); // Clear Screen

		WriteToLCD(cur_pos, 0); // Move to current position
		WriteToLCD(0x3E, 1); // 2

		if(cur_pos==0x8F){
			cur_pos = 0x80;
			return;
		}

		cur_pos += 0x1;

		WriteToLCD(cur_pos, 0); // Move to current position
		WriteToLCD(0x3C, 1); // 4
	}
	else if(mode==1){
		if(cur_digit>total_num) cur_digit = 0;

		WriteToLCD(0x80+cur_digit, 0);
		WriteToLCD(str_arr[cur_digit], 1);
		cur_digit++;
	}
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
	SysTick->LOAD = (uint32_t)400000*3-1; //think 1 think
	// CLKSOURCE processor clock
	SET_REG(SysTick->CTRL,SysTick_CTRL_CLKSOURCE_Msk,SysTick_CTRL_CLKSOURCE_Msk);
}

void init_GPIO(){
	RCC->AHB2ENR = RCC->AHB2ENR|0x6; //B

	//Set PB0-8,13,14,15 as output mode
	GPIOB->MODER= GPIOB->MODER&0x03FC0000;
	GPIOB->MODER= GPIOB->MODER|0x54015555;

	//Set PC13 as input mode
	GPIOC->MODER= GPIOC->MODER&0xF3FFFFFF;

}

int main(){

	for(total_num=1; (str_arr[total_num-1]!='\0')&&(total_num<=16); total_num++);
	total_num-=2;
	cur_digit = 0;

	init_GPIO();
	init_LCD();
	Init_SysTick();
	SystemClock_Config_source();

	while(1);
}
