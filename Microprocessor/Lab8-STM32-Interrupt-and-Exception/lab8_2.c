#include <stdio.h>
#include <stdlib.h>
#include "stm32l476xx.h"
#include "system_stm32l4xx.h"
#include "core_cmSimd.h"
#include "core_cmInstr.h"
#include "core_cmFunc.h"
#include "core_cm4.h"
#include "cmsis_gcc.h"
#define X0 0b100000
#define X1 0b1000000
#define X2 0b10000000
#define X3 0b100000000
#define Y0 0b100000
#define Y1 0b1000000
#define Y2 0b10000000
#define Y3 0b100000000

unsigned int x_pin[4] = {X0, X1, X2, X3};
unsigned int y_pin[4] = {Y0, Y1, Y2, Y3};
int keypad_value[4][4]={{1,2,3,10},{4,5,6,11},{7,8,9,12},{15,0,14,13}};

extern void Delay();
#define SET_REG(REG,SELECT,VAL){((REG)=((REG)&(~(SELECT)))|(VAL));};


void EXTI_config(){
	// SYSCFGEN
	SET_REG(RCC->APB2ENR,1,1);

	// set GPIO PC5,6,7,8 as EXTIx source
	SET_REG(SYSCFG->EXTICR[1],0xF0,SYSCFG_EXTICR2_EXTI5_PC);
	SET_REG(SYSCFG->EXTICR[1],0xF00,SYSCFG_EXTICR2_EXTI6_PC);
	SET_REG(SYSCFG->EXTICR[1],0xF000,SYSCFG_EXTICR2_EXTI7_PC);
	SET_REG(SYSCFG->EXTICR[2],0xF,SYSCFG_EXTICR3_EXTI8_PC);

	// select rising edge
	SET_REG(EXTI->RTSR1,EXTI_RTSR1_RT5|EXTI_RTSR1_RT6|EXTI_RTSR1_RT7|EXTI_RTSR1_RT8,EXTI_RTSR1_RT5|EXTI_RTSR1_RT6|EXTI_RTSR1_RT7|EXTI_RTSR1_RT8);
	SET_REG(EXTI->FTSR1,EXTI_FTSR1_FT5|EXTI_FTSR1_FT6|EXTI_FTSR1_FT7|EXTI_FTSR1_FT8,0);

	// enable this interrupt
	SET_REG(EXTI->IMR1,EXTI_IMR1_IM5|EXTI_IMR1_IM6|EXTI_IMR1_IM7|EXTI_IMR1_IM8,EXTI_IMR1_IM5|EXTI_IMR1_IM6|EXTI_IMR1_IM7|EXTI_IMR1_IM8);


	// pending flag??
	SET_REG(EXTI->PR1,EXTI_PR1_PIF5|EXTI_PR1_PIF6|EXTI_PR1_PIF7|EXTI_PR1_PIF8|EXTI_PR1_PIF9,EXTI_PR1_PIF5|EXTI_PR1_PIF6|EXTI_PR1_PIF7|EXTI_PR1_PIF8|EXTI_PR1_PIF9);


}

void NVIC_config(){


	// enable IRQ
	//NVIC_EnableIRQ(EXTI9_5_IRQn);
	SET_REG(NVIC->ISER[0],(1<<23),(1<<23));

}

void EXTI9_5_IRQHandler(void){
	uint32_t col_deter = GPIOA->ODR >> 5;
	uint32_t row_deter = EXTI->PR1 >> 5;

	int row = 0;
	uint32_t row_temp = row_deter;
	for(; row<4; row++){
		row_temp &= 1;
		if(row_temp==1) break;
		row_temp = (row_deter>>(row+1));
	}

	int col = 0;
	uint32_t col_temp = col_deter;
	for(; col<4; col++){
		col_temp &= 1;
		if(col_temp==1) break;
		col_temp = (col_deter>>(col+1));


	}

	int count = keypad_value[row][col]; //press value
	for(int i=0; i<count; i++){
		GPIOB->BSRR = (1<<10);
		Delay();
		GPIOB->BRR = (1<<10);
		Delay();
	}
	// interrupt turn off
	SET_REG(EXTI->PR1,EXTI_PR1_PIF5|EXTI_PR1_PIF6|EXTI_PR1_PIF7|EXTI_PR1_PIF8|EXTI_PR1_PIF9,EXTI_PR1_PIF5|EXTI_PR1_PIF6|EXTI_PR1_PIF7|EXTI_PR1_PIF8|EXTI_PR1_PIF9);//clear PR1
	//sdfjkljklsdfjklsdfksdfjlfjklsdjklsdffjklsdklsdfjjkldfssfkldjsdfjkljsdfkllfjksdsdfjklfjklsdjklsdf
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

		//GPIOA->ODR=GPIOA->ODR|0b1111<<5;

		// SET LED gpio OUTPUT, PB10 //
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

void keypad_scan(){
	unsigned int flag_keypad,flag_debounce;
	while(1){
		//flag_keypad=GPIOC->IDR&0b1111<<5;
		//if(flag_keypad!=0){
			/*int k=45000;
			while(k!=0){
				flag_debounce=GPIOC->IDR&0b1111<<5;
				k--;
			}
			if(flag_debounce!=0){*/
				int position_c;
				for(int i=0;i<4;i++){ //scan keypad from first column
					position_c=i+5;
					//set PA5,6,7,8(column) low and set pin high from PA5
					GPIOA->ODR=(GPIOA->ODR&0xFFFFFE1F)|1<<position_c;
					int position_r;
					unsigned int flag_keypad_r;
					/*for(int j=0;j<4;j++){ //read input from first row
						position_r=j+5;
						flag_keypad_r=GPIOC->IDR&(1<<position_r);
						if(flag_keypad_r!=0){
							// interrupt enable
						}
					}*/
				}
			//}
		//}

		//GPIOA->ODR=GPIOA->ODR|0b1111<<5; //set PA5,6,7,8(column) high
	}
}


int main(){
	NVIC_config();
	EXTI_config();
	init_GPIO();

	keypad_scan();

	return 0;
}
