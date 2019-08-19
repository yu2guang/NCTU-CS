#include <stdio.h>
#include <stdlib.h>
#include "stm32l476xx.h"
#include "system_stm32l4xx.h"
#include "core_cmSimd.h"
#include "core_cmInstr.h"
#include "core_cmFunc.h"
#include "core_cm4.h"
#include "cmsis_gcc.h"
#include "ref.h"
#include <string.h>

#define SET_REG(REG,SELECT,VAL){((REG)=((REG)&(~(SELECT)))|(VAL));};
char arr[100]= "pinyugly";

void SystemInit(void) {
#define    __FPU_PRESENT       1U
    //#if (__FPU_USED == 1)
    SCB->CPACR |= (3UL << 20) | (3UL << 22);
    __DSB();
    __ISB();
    //#endif
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
}

void ADC_transmit(int Voltage){
    char trans_V[5];
    sprintf(trans_V, "%d", Voltage);
    print_str(trans_V);
}

void GPIO_Init(void) {
    RCC->AHB2ENR |= (RCC_AHB2ENR_GPIOAEN | RCC_AHB2ENR_GPIOBEN | RCC_AHB2ENR_GPIOCEN);
    
    /* LED */
    SET_REG(GPIOA->MODER,0x00000C00,0x400);
    
    /* UART */
    // AF MODE: PC10, 11
    SET_REG(GPIOC->MODER,0x00F00000,0x00A00000);
    // AFRH: AFSEL10, 11; AF7
    SET_REG(GPIOC->AFR[1],0x0000FF00,0x00007700);
    
    /* BUTTON */
    SET_REG(GPIOC->MODER,0x0C000000,0);
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
    UART_Transmit('\r');
    UART_Transmit('\n');
    for(int j=0; j<strlen(str); j++){
        UART_Transmit(str[j]);
    }
}

void LED_switch(int value){
    if(value==1){
        GPIOA->ODR &= ~(1<<5);
        GPIOA->ODR |= (1<<5);
    }
    else{
        GPIOA->ODR &= ~(1<<5);
    }
}

void Init_SysTick(){
    // enable SysTick
    SET_REG(SysTick->CTRL,SysTick_CTRL_ENABLE_Msk,SysTick_CTRL_ENABLE_Msk);
    // set reload register
    SET_REG(SysTick->CTRL,SysTick_CTRL_TICKINT_Msk,SysTick_CTRL_TICKINT_Msk); // exception enable
    SET_REG(SysTick->CTRL,SysTick_CTRL_COUNTFLAG_Msk,SysTick_CTRL_COUNTFLAG_Msk); // VAL=0, reload
    // Load the SysTick Counter Value
    SysTick->LOAD = (uint32_t)2000000-1;
    // CLKSOURCE processor clock
    SET_REG(SysTick->CTRL,SysTick_CTRL_CLKSOURCE_Msk,SysTick_CTRL_CLKSOURCE_Msk);
}

void SysTick_Handler(void){
    uint32_t ADC1ConvertedValue;
    int ADC1ConvertedVoltage;
    // get ADC1 converted data //read ADC_DR will let EOC flag be cleared
    ADC1ConvertedValue = ADC1->DR;
    // compute the V
    ADC1ConvertedVoltage = (int)((int)ADC1ConvertedValue*3300)/4096;
    
    GPIOA->ODR ^= (0b1<<5);
    
    // next line
    // UART_Transmit('A');
    ADC_transmit(ADC1ConvertedVoltage);
    UART_Transmit('\r');
    UART_Transmit('\n');
    
}

void light(){
    Init_SysTick();
    
    while(1){
        if(UART_Receive(1)=='q'){
            UART_Transmit('q');
            SET_REG(RCC->APB2ENR,1,0);
            SET_REG(SysTick->CTRL,1,0);
            break;
        }
    }
}

int main(void){
    
    GPIO_Init();
    USART3_Init();
    ADC_Config();
    delay_us(10);
    
    
    char input[100];
    int num;
    for(int i=0;; i++){
        if(i!=0){ // next line
            UART_Transmit('\r');
            UART_Transmit('\n');
        }
        UART_Transmit('>');
        for(num=0; num<100; ){
            input[num] = UART_Receive(0);
            if(input[num]=='\r') break;
            UART_Transmit(input[num]);
            if(input[num]=='\b'){ // backspace
                UART_Transmit(' ');
                UART_Transmit('\b');
                num--;
                input[num] = '\r';
            }
            else{
                num++;
            }
        }
        
        /* determine input */
        // nothing
        if(strncmp(input,"\r",1)==0){
            continue;
        }
        
        if(strncmp(input,"showid",6)==0){
            print_str("0516054");
        }
        else if(strncmp(input,"light",5)==0){
            light();
        }
        else if(strncmp(input,"led on",6)==0){
            LED_switch(1);
        }
        else if(strncmp(input,"led off",7)==0){
            LED_switch(0);
        }
        else if(strncmp(input,"led show",8)==0){
            if((GPIOA->ODR>>5)&1==1){
                print_str("bling bling bling");
            }
            else{
                print_str("wooooo");
            }
        }
        else{
            print_str("Zooooooooah!!");
            continue;
        }
    }
    
    
    return 0;
}
