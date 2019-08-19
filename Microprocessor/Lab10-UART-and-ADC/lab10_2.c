#include <stdio.h>
#include <stdlib.h>
#include "stm32l476xx.h"
#include "system_stm32l4xx.h"
#include "core_cmSimd.h"
#include "core_cmInstr.h"
#include "core_cmFunc.h"
#include "core_cm4.h"
#include "cmsis_gcc.h"
#include <string.h>
#include <ref.h>

#define SET_REG(REG,SELECT,VAL){((REG)=((REG)&(~(SELECT)))|(VAL));};
int update=0,change=1;


void SystemInit(void) {
#define    __FPU_PRESENT       1U
    //#if (__FPU_USED == 1)
    SCB->CPACR |= (3UL << 20) | (3UL << 22);
    __DSB();
    __ISB();
    //#endif
}

void GPIO_Init(void) {
    
	RCC->AHB2ENR |= (RCC_AHB2ENR_GPIOAEN | RCC_AHB2ENR_GPIOBEN | RCC_AHB2ENR_GPIOCEN);
	/* UART */
	// AF MODE: PC10, 11
	SET_REG(GPIOC->MODER,0x00F00000,0x00A00000);
	// AFRH: AFSEL10, 11; AF7
	SET_REG(GPIOC->AFR[1],0x0000FF00,0x00007700);

	/* BUTTON */
	SET_REG(GPIOC->MODER,0x0C000000,0);
}


void USART3_Init(void) {

	/* Enable clock for USART3 */
	RCC->APB1ENR1 |= RCC_APB1ENR1_USART3EN;

	/* CR1 */
	//USART disable
	SET_REG(USART3->CR1,USART_CR1_UE,0);
	//M bit 00->8bit
	SET_REG(USART3->CR1,USART_CR1_M,0);
	//oversampling=16
	SET_REG(USART3->CR1,USART_CR1_OVER8,0);
	//0b110100001);//Baud=fck/USARTDIV  //USARTDIV=417    //fck=4M    //Baud=9600
	SET_REG(USART3->BRR,USART_BRR_DIV_MANTISSA,4000000L/9600L);

	/* CR2 */
	// 1-bit stop
	SET_REG(USART3->CR2, USART_CR2_STOP, 0x0);

	/* Enable UART */
	USART3->CR1 |= (USART_CR1_UE);
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

void print_str(char str[]){
    UART_Transmit('\r');
    UART_Transmit('\n');
    for(int j=0; j<strlen(str); j++){
        UART_Transmit(str[j]);
    }
}

void NVIC_config(){
	// enable IRQ
	NVIC_EnableIRQ(ADC1_2_IRQn);
}

void ADC_Config(void)
{
	/* initial PC2 in analog mode */
	// PC2, ADC123_IN3
	SET_REG(RCC->AHB2ENR,RCC_AHB2ENR_GPIOCEN,RCC_AHB2ENR_GPIOCEN);
	SET_REG(GPIOC->MODER,0x30,0x30); //channel 3
	SET_REG(GPIOC->ASCR,(1<<2),(1<<2)); // PC2, analog

	/* ADC configuration */
	// bus enable
	SET_REG(RCC->AHB2ENR,RCC_AHB2ENR_ADCEN,RCC_AHB2ENR_ADCEN);

	// continuous convert mode
	ADC1->CFGR |= ADC_CFGR_CONT;
	// 12-bit data resolution: 00
	ADC1->CFGR &= ~ADC_CFGR_RES;
	// right data alignment: 0
	ADC1->CFGR &= ~ADC_CFGR_ALIGN;
	// OVRMOD = 1, or the new data will be discard
	SET_REG(ADC1->CFGR,ADC_CFGR_OVRMOD,ADC_CFGR_OVRMOD);
	// interrupt can generate
	SET_REG(ADC1->IER,ADC_IER_EOCIE,ADC_IER_EOCIE);

	/* ADC1 regular channel 3 configuration */
	// SQR1=3, channel 3
	ADC1->SQR1 |= (ADC_SQR1_SQ1_1 | ADC_SQR1_SQ1_0);
	// 0000: 1 conversion
	ADC1->SQR1 &= ~ADC_SQR1_L;
	// sampling rate
	ADC1->SMPR1 |= ADC_SMPR1_SMP3_1 | ADC_SMPR1_SMP3_0;

	// no save energy
	SET_REG(ADC1->CR,ADC_CR_DEEPPWD,0);
	// ADC V enable
	SET_REG(ADC1->CR,ADC_CR_ADVREGEN,ADC_CR_ADVREGEN);
	delay_us(10);
	// enable ADC1
	SET_REG(ADC1->CR,ADC_CR_ADEN,ADC_CR_ADEN);
	// CLK src
	SET_REG(ADC123_COMMON->CCR,(0b11<<15),(0b11<<15));

	// wait for ADRDY
	while(!ADC1->ISR & ADC_ISR_ADRDY);

	// start ADC1 software conversion
	SET_REG(ADC1->CR,ADC_CR_ADSTART,ADC_CR_ADSTART);
    
	/* calibration procedure */ //no need, but will make ADC more precise
	/*//ADC V enable
	SET_REG(ADC1->CR,ADC_CR_ADVREGEN,ADC_CR_ADVREGEN);
	delay_us(10);
	//Writing ADCAL will launch a calibration in single-ended inputs mode. ???
	SET_REG(ADC1->CR,ADC_CR_ADCALDIF,0);
	//start calibration
	SET_REG(ADC1->CR,ADC_CR_ADCAL,ADC_CR_ADCAL);
	//read at one means that a calibration in progress
	while(ADC1->CR & ADC_CR_ADCAL);//wait until calibration done
	int calibration_value=ADC1->CALFACT;*/
}

void ADC1_2_IRQHandler(void){

	/* After finish conversion, go into handler.
	 * show result of conversion  */

	uint32_t ADC1ConvertedValue;
	int ADC1ConvertedVoltage;
	// get ADC1 converted data //read ADC_DR will let EOC flag be cleared
	ADC1ConvertedValue = ADC1->DR;
	// compute the V
	ADC1ConvertedVoltage = (int)((int)ADC1ConvertedValue*3300)/4096;
    

    if((GPIOC->IDR>>13)==1){
		change=1;
	}
	if(change==1){
		if(update==0){
            while(1){
                if((GPIOC->IDR>>13)==0){
                    update=1;
                    break;
                }
            }
        }
        else if(update==1){
            ADC_transmit(ADC1ConvertedVoltage);
            change=0;
            update=0;
        }
	}
}

void ADC_transmit(int Voltage){
	char trans_V[5];
	sprintf(trans_V,"%d",Voltage);
	print_str(trans_V);
}

int main(){
	GPIO_Init();
	USART3_Init();
	NVIC_config();
	ADC_Config();

	while(1);
}
