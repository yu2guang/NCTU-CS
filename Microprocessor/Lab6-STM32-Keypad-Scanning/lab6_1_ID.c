#include <stdlib.h>
#include <stdio.h>

extern void GPIO_init();
extern void max7219_init();
extern void max7219Send();
int arr[10]={4,5,0,6,1,5,0};

void display(int num){
	int addr,data;
	addr=num+1;
	data=arr[num];
	max7219Send(addr,data);
}

int main(void){
	GPIO_init();
	max7219_init();
	while(1){
		for(int i=0; i<7; i++){
			display(i);
		}
	}
	return 0;
}
