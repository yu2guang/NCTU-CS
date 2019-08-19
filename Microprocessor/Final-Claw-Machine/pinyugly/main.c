#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stm32l476xx.h"
#include "system_stm32l4xx.h"
#include "core_cmSimd.h"
#include "core_cmInstr.h"
#include "core_cmFunc.h"
#include "core_cm4.h"
#include "cmsis_gcc.h"
#include "ref.h"

#define SET_REG(REG,SELECT,VAL){((REG)=((REG)&(~(SELECT)))|(VAL));};

#define motorOpCLSec 5
#define motorUpDownSec 5
#define motorFroBackSec 5
#define motorLRSec 5
#define colTotal 1442
#define rowTotal 1540
#define verticalTotal 821
#define openTotal 217
#define closeTotal 398
#define colEnd 5
#define rowEnd 5

void SystemInit(void) {
#define    __FPU_PRESENT       1U
    //#if (__FPU_USED == 1)
    SCB->CPACR |= (3UL << 20) | (3UL << 22);
    __DSB();
    __ISB();
    //#endif
}

/* initial */
void init_GPIO();
void USART3_Init(void);
void var_initial();
void Init_SysTick();
int sec_cur = 1;
int row_clean = 0;
int col_clean = 0;

/* UART */
void scan_uart();
void UART_Transmit(char one);
int UART_Receive(int isSystik);
void print_str(char str[]);

/* motor */
void motor_open_close(int close);
void motor_up_down(int down);
void motor_front_back(int back);
void motor_left_right(int right);

/* finish */
void gatcha();
int end = 0;
int row = 0, col = 0;


void init_GPIO(){
	RCC->AHB2ENR = RCC->AHB2ENR|0x7;

	/* motor */
	// output mode
	// PA5~8: motor_open_close
	// PA9~12: motor_up_down
	GPIOA->MODER= GPIOA->MODER&0xFD5557FF;
	// PB5~8: motor_front_back
	// PB9~12: motor_left_right
	GPIOB->MODER= GPIOB->MODER&0xFD5557FF;

	/* UART */
	// AF mode: PC10, 11
	SET_REG(GPIOC->MODER,0x00F00000,0x00A00000);
	// AFRH: AFSEL10, 11; AF7
	SET_REG(GPIOC->AFR[1],0x0000FF00,0x00007700);

	/* user button */
	// input mode: PC13
	GPIOC->MODER= GPIOC->MODER&0xF3FFFFFF;
}

void USART3_Init(void) {
    /* Enable clock for USART */
    RCC->APB1ENR1 |= RCC_APB1ENR1_USART3EN;

    /* CR1 */
    // USART disable
    SET_REG(USART3->CR1,USART_CR1_UE,0);
    // M bit 00->8bit
    SET_REG(USART3->CR1,USART_CR1_M,0);
    // oversampling=16
    SET_REG(USART3->CR1,USART_CR1_OVER8,0);
    //Baud=fck/USARTDIV  //USARTDIV=417    //fck=4M    //Baud=9600
    SET_REG(USART3->BRR,USART_BRR_DIV_MANTISSA,4000000L/9600L);

    /* CR2 */
    // 1-bit stop bit
    SET_REG(USART3->CR2, USART_CR2_STOP, 0x0);

    /* Enable UART */
    USART3->CR1 |= (USART_CR1_UE);
}

void var_initial(){
	//sec_cur = rand()%(15-3)+3;
	//sec_cur = 60;
	sec_cur = 1000;
	end = 0;
	row = 0;
	col = 0;
	row_clean = 0;
	col_clean = 0;
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
}

void SysTick_Handler(void){
	if(sec_cur>0){
		sec_cur--;
		return;
	}

	end = 1;
	gatcha();
}

void scan_uart(){
	/* welcome */
	char trans_V[5];
	print_str("------------------------------\n\r\n\r");
	print_str("Welcome :D\n\r");
	print_str("You have ");
	sprintf(trans_V, "%d", sec_cur);
	print_str(trans_V);
	print_str(" sec\n\r");
	print_str("START!!\n\r\n\r");
	//print_str(">\27\e");

	/* scan user command */
	char command;
	while(1){
		if(end==1) break;

		command = UART_Receive(1);
		if(command=='\r'){
			gatcha();
		    break;
		}
		else if(command=='s'){
			if(col_clean==1){
				print_str("\b\r                          \r\n");
				col_clean = 0;
				col++;
			}
			else if(row_clean==1){
				print_str("\b\r                          \r\n");
				row_clean = 0;
			}

			print_str("front\r");
			if(col>0){
				col--;
				motor_front_back(0);
			}
			else{
				print_str("\b\rNo more front steps!!\r\n");
				col_clean = 1;
			}
		}
		else if(command=='w'){
			if(col_clean==1){
				print_str("\b\r                          \r\n");
				col_clean = 0;
				col++;
			}
			else if(row_clean==1){
				print_str("\b\r                          \r\n");
				row_clean = 0;
			}

			print_str("back \r");
			if(col<colEnd){
				motor_front_back(1);
				col++;
			}
			else{
				print_str("\b\rNo more back steps!!\r\n");
				col_clean = 1;
			}
		}
		else if(command=='a'){
			if(row_clean==1){
				print_str("\b\r                          \r\n");
				row_clean = 0;
				row--;
			}
			else if(col_clean==1){
				print_str("\b\r                          \r\n");
				col_clean = 0;
			}

			print_str("left \r");
			int a;
			if(row>0){
				motor_left_right(0);
				row--;
				a = row;
			}
			else{
				print_str("\b\rNo more left steps!!\r\n");
				row_clean = 1;
			}
		}
		else if(command=='d'){
			if(row_clean==1){
				print_str("\b\r                          \r\n");
				row_clean = 0;
				row++;
			}
			else if(col_clean==1){
				print_str("\b\r                          \r\n");
				col_clean = 0;
			}

			print_str("right\r");
			if(row<rowEnd){
				motor_left_right(1);
				row++;
			}
			else{
				print_str("\b\rNo more right steps!!\r\n");
				row_clean = 1;
			}
		}
		else if(command=='o'){
			print_str("open \r");
			motor_open_close(0);
		}
		else if(command=='p'){
			print_str("close\r");
			motor_open_close(1);
		}
		else if(command=='u'){
			print_str("upp  \r");
			motor_up_down(0);
		}
		else if(command=='j'){
					print_str("down  \r");
					motor_up_down(1);
				}
	}
}

void UART_Transmit(char one){

    SET_REG(USART3->CR1,USART_CR1_TE,USART_CR1_TE);
    int TE_bit;
    while(1){
        TE_bit = (USART3->CR1 >> 3) & 1;
        if(TE_bit==1) break;
    }
    SET_REG(USART3->ISR,USART_ISR_TXE, USART_ISR_TXE);
    USART3->TDR = one;

    int TC_bit;
    while (1){
        TC_bit = (USART3->ISR >> 6) & 1;
        if(TC_bit==1) break;
    }
}

int UART_Receive(int isSystik){
    // enable receiver
    SET_REG(USART3->CR1,USART_CR1_RE,USART_CR1_RE);

    // wait until RXNE set
    int RXNE_bit = (USART3->ISR >> 5) & 1;
    // systik time
    if(isSystik==1){
        if(RXNE_bit!=1){
            return 0;
        }
    }
    while (1){
        RXNE_bit = (USART3->ISR >> 5) & 1;
        if(RXNE_bit==1) break;
    }

    // read data from RDR
    return USART3->RDR;
}

void print_str(char str[]){
    for(int j=0; j<strlen(str); j++){
        UART_Transmit(str[j]);
    }
}

void motor_open_close(int close){
	/* PA5~8 */
	// take turns to connect GND
	SET_REG(GPIOA->ODR,(0b1111<<5),(0b1111<<5));

	int push;
	int test = 0;
	if(close==1){
		for(int i=0; i<closeTotal; i++){
		//for(int i=0; ; i++){
			test++;
			SET_REG(GPIOA->ODR,(1<<5),0);
			delay_ms(motorOpCLSec);
			GPIOA->ODR |= 1<<5;

			//push = (GPIOC->IDR>>13)&1;
				//							if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<6),0);
			delay_ms(motorOpCLSec);
			GPIOA->ODR |= 1<<6;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<7),0);
			delay_ms(motorOpCLSec);
			GPIOA->ODR |= 1<<7;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<8),0);
			delay_ms(motorOpCLSec);
			GPIOA->ODR |= 1<<8;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;
		}
	}
	else{
		for(int i=0; i<openTotal; i++){
		//for(int i=0; ; i++){
			test++;
			SET_REG(GPIOA->ODR,(1<<8),0);
			delay_ms(motorOpCLSec);
			GPIOA->ODR |= 1<<8;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<7),0);
			delay_ms(motorOpCLSec);
			GPIOA->ODR |= 1<<7;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<6),0);
			delay_ms(motorOpCLSec);
			GPIOA->ODR |= 1<<6;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<5),0);
			delay_ms(motorOpCLSec);
			GPIOA->ODR |= 1<<5;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;
		}
	}

	int stop;
}

void motor_up_down(int down){
	/* PA9~12 */
	// take turns to connect GND
	SET_REG(GPIOA->ODR,(0b1111<<9),(0b1111<<9));

	int push;
	int test = 0;
	if(down==0){
		for(int i=0; i<verticalTotal; i++){
		//while(1){
			test++;
			SET_REG(GPIOA->ODR,(1<<9),0);
			delay_ms(motorUpDownSec);
			GPIOA->ODR |= 1<<9;

			push = (GPIOC->IDR>>13)&1;
											if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<10),0);
			delay_ms(motorUpDownSec);
			GPIOA->ODR |= 1<<10;

			push = (GPIOC->IDR>>13)&1;
											if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<11),0);
			delay_ms(motorUpDownSec);
			GPIOA->ODR |= 1<<11;

			push = (GPIOC->IDR>>13)&1;
											if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<12),0);
			delay_ms(motorUpDownSec);
			GPIOA->ODR |= 1<<12;

			push = (GPIOC->IDR>>13)&1;
											if(push==0) break;
		}
	}
	else{
		for(int i=0; i<verticalTotal; i++){
		//while(1){
			test++;
			SET_REG(GPIOA->ODR,(1<<12),0);
			delay_ms(motorUpDownSec);
			GPIOA->ODR |= 1<<12;

			push = (GPIOC->IDR>>13)&1;
														if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<11),0);
			delay_ms(motorUpDownSec);
			GPIOA->ODR |= 1<<11;

			push = (GPIOC->IDR>>13)&1;
														if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<10),0);
			delay_ms(motorUpDownSec);
			GPIOA->ODR |= 1<<10;

			push = (GPIOC->IDR>>13)&1;
														if(push==0) break;

			SET_REG(GPIOA->ODR,(1<<9),0);
			delay_ms(motorUpDownSec);
			GPIOA->ODR |= 1<<9;

			push = (GPIOC->IDR>>13)&1;
														if(push==0) break;
		}
	}

	int stop = test;

}

void motor_front_back(int back){
	/* PB5~8 */
	// take turns to connect GND
	SET_REG(GPIOB->ODR,(0b1111<<5),(0b1111<<5));

	int test = 0;
	int push;
	if(back==1){
		for(int i=0; i<(colTotal/colEnd); i++){
		//while(1){
			test++;
			SET_REG(GPIOB->ODR,(1<<8),0);
			delay_ms(motorFroBackSec);
			GPIOB->ODR |= 1<<8;

			//push = (GPIOC->IDR>>13)&1;
			//					if(push==0) break;

			SET_REG(GPIOB->ODR,(1<<7),0);
			delay_ms(motorFroBackSec);
			GPIOB->ODR |= 1<<7;

			//push = (GPIOC->IDR>>13)&1;
			//					if(push==0) break;

			SET_REG(GPIOB->ODR,(1<<6),0);
			delay_ms(motorFroBackSec);
			GPIOB->ODR |= 1<<6;

			//push = (GPIOC->IDR>>13)&1;
			//					if(push==0) break;

			SET_REG(GPIOB->ODR,(1<<5),0);
			delay_ms(motorFroBackSec);
			GPIOB->ODR |= 1<<5;

			//push = (GPIOC->IDR>>13)&1;
			//					if(push==0) break;
		}
	}
	else{
		for(int i=0; i<(colTotal/colEnd); i++){
		//while(1){
			SET_REG(GPIOB->ODR,(1<<5),0);
			delay_ms(motorFroBackSec);
			GPIOB->ODR |= 1<<5;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;

			SET_REG(GPIOB->ODR,(1<<6),0);
			delay_ms(motorFroBackSec);
			GPIOB->ODR |= 1<<6;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;

			SET_REG(GPIOB->ODR,(1<<7),0);
			delay_ms(motorFroBackSec);
			GPIOB->ODR |= 1<<7;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;

			SET_REG(GPIOB->ODR,(1<<8),0);
			delay_ms(motorFroBackSec);
			GPIOB->ODR |= 1<<8;

			//push = (GPIOC->IDR>>13)&1;
			//								if(push==0) break;
		}
	}

	int a = test;
}

void motor_left_right(int right){
	/* PB9~12 */
	// take turns to connect GND
	SET_REG(GPIOB->ODR,(0b1111<<9),(0b1111<<9));

	int test = 0;
	int push;
	if(right==1){
		for(int i=0; i<(rowTotal/rowEnd); i++){
		//while(1){
			test++;
			SET_REG(GPIOB->ODR,(1<<9),0);
			delay_ms(motorLRSec);
			GPIOB->ODR |= 1<<9;

			//push = (GPIOC->IDR>>13)&1;
			//											if(push==0) break;

			SET_REG(GPIOB->ODR,(1<<10),0);
			delay_ms(motorLRSec);
			GPIOB->ODR |= 1<<10;

			//push = (GPIOC->IDR>>13)&1;
			//											if(push==0) break;

			SET_REG(GPIOB->ODR,(1<<11),0);
			delay_ms(motorLRSec);
			GPIOB->ODR |= 1<<11;

			//push = (GPIOC->IDR>>13)&1;
			//											if(push==0) break;

			SET_REG(GPIOB->ODR,(1<<12),0);
			delay_ms(motorLRSec);
			GPIOB->ODR |= 1<<12;

			//push = (GPIOC->IDR>>13)&1;
			//											if(push==0) break;
		}
	}
	else{
		for(int i=0; i<(rowTotal/rowEnd); i++){
		//while(1){
			SET_REG(GPIOB->ODR,(1<<12),0);
			delay_ms(motorLRSec);
			GPIOB->ODR |= 1<<12;

			SET_REG(GPIOB->ODR,(1<<11),0);
			delay_ms(motorLRSec);
			GPIOB->ODR |= 1<<11;

			SET_REG(GPIOB->ODR,(1<<10),0);
			delay_ms(motorLRSec);
			GPIOB->ODR |= 1<<10;

			SET_REG(GPIOB->ODR,(1<<9),0);
			delay_ms(motorLRSec);
			GPIOB->ODR |= 1<<9;

		}
	}

	int stop = test;

}

void gatcha(){
	print_str("\n\r~~pinyugly~~\n\r\n\r");
	/* disable systick */
	SET_REG(SysTick->CTRL,1,0);

	/* start catching */
	motor_up_down(1);
	motor_open_close(1);
	motor_up_down(0);

	/* back to START position */
	// col
	for(int i=0; i<col; i++){
		motor_front_back(0);
	}
	//row
	for(int i=0; i<col; i++){
		motor_left_right(0);
	}

	/* release */
	motor_open_close(0);
}

int main(){
    // initial
	init_GPIO();
    USART3_Init();

	int push;
    while(1){
        // wait until user starts
        while(1){
            push = (GPIOC->IDR>>13)&1;
            if(push==0) break;
        }

        var_initial();

        // start to count down
        Init_SysTick();

        // scan user command
        scan_uart();
    }

}
