#include "stm32l476xx.h"
#include <stdlib.h>
#include <stdio.h>
//TODO: define your gpio pin
#define X0 0b100000
#define X1 0b1000000
#define X2 0b10000000
#define X3 0b100000000
#define Y0 0b100000
#define Y1 0b1000000
#define Y2 0b10000000
#define Y3 0b100000000

extern void GPIO_init();
extern void max7219_init();
extern void max7219Send();

unsigned int x_pin[4] = {X0, X1, X2, X3};
unsigned int y_pin[4] = {Y0, Y1, Y2, Y3};
int arr[4][4]={{1,2,3,10},{4,5,6,11},{7,8,9,12},{15,0,14,13}};
int store=0;

char keypad_scan();

void display(int num){
	int addr,data;
	//int deter = (num/10);
	int deter=-1,num_temp;
	num_temp=num;
	while(1){
		num_temp=(num_temp/10);
		deter++;
		if(num_temp==0)break;
	}

	if(deter==0){
		max7219Send(11,0);
		addr=1;
		data=num;
		max7219Send(addr,data);
	}else{
		max7219Send(11,deter);
		int temp=num;
		deter++;
		for(int i=1;i<=deter;i++){
			addr=i;
			data=temp%10;
			temp=temp/10;
			max7219Send(addr,data);
		}
	}
}

/* TODO: initial keypad gpio pin, X as output and Y as input */
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

	// SET keypad gpio INPUT //
	//Set PC5,6,7,8 as INPUT mode
	GPIOC->MODER=GPIOC->MODER&0xFFFC03FF;
	//set PC5,6,7,8 is Pull-down input
	GPIOC->PUPDR=GPIOC->PUPDR|0x2A800;
	//Set PC5,6,7,8 as medium speed mode
	GPIOC->OSPEEDR=GPIOC->OSPEEDR|0x15400;

}
int main(void) {
	GPIO_init();
	max7219_init();
	keypad_init();
    
	display(store);
	while(1) {
		keypad_scan();

	}
}


/* TODO: scan keypad value return: >=0: key pressedvalue -1: no keypress */


char keypad_scan(){
	/*for(int j=0; j<4; j++){
		for(int k=0;k<4;k++){
			GPIOA->BRR =x_pin[k];//set bit as low
		}
		GPIOA->BSRR = x_pin[j];//set bit as high

		for(int i=0; i<4; i++){
			int temp = ((GPIOC->IDR)>>y_pin[i]) && 1;
			if(temp)
				display(arr[i][j]);
		}
	}*/
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
						flag_keypad_r=GPIOC->IDR&1<<position_r;
						if(flag_keypad_r!=0){
							if(arr[j][i]==15||arr[j][i]==14){
                                store=0; //將store歸零
								display(store); //display store
								continue; //進入下一個迴圈
							}

							if((store+arr[j][i])>99999999){
								display(store); //display 原來的數字不更改
								continue; //進入下一個迴圈
							}
							else{
								store=store+arr[j][i]; //將store加上現在按到的按鍵值
								display(store); //display store
							}

						}
					}
				}
			}
		}
		GPIOA->ODR=GPIOA->ODR|0b1111<<5; //set PA5,6,7,8(column) high
	}
	return 'a';
}
