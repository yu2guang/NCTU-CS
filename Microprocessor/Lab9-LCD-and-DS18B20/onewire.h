#ifndef ONEWIRE_H_
#define ONEWIRE_H_

#include "stm32l476xx.h"

typedef struct {
	GPIO_TypeDef* GPIOx;           /*!< GPIOx port to be used for I/O functions */
	uint32_t GPIO_Pin;             /*!< GPIO Pin to be used for I/O functions */
} OneWire_t;

extern void Delay_us();
extern void Delay_10ms();
#define SET_REG(REG,SELECT,VAL){((REG)=((REG)&(~(SELECT)))|(VAL));};

void OneWire_Init(OneWire_t* OneWireStruct, GPIO_TypeDef* GPIOx, uint32_t GPIO_Pin);
void OneWire_SkipROM(OneWire_t* OneWireStruct);
uint8_t OneWire_Reset(OneWire_t* OneWireStruct);
uint8_t OneWire_ReadByte(OneWire_t* OneWireStruct);
void OneWire_WriteByte(OneWire_t* OneWireStruct, uint8_t byte);
void OneWire_WriteBit(OneWire_t* OneWireStruct, uint8_t bit);
uint8_t OneWire_ReadBit(OneWire_t* OneWireStruct);

void ONEWIRE_OUTPUT(OneWire_t* OneWireStruct);
void ONEWIRE_INPUT(OneWire_t* OneWireStruct);
void ONEWIRE_LOW(OneWire_t* OneWireStruct);
void ONEWIRE_DELAY_10ms(int sec);
void ONEWIRE_DELAY_us(int sec);

void write_three_things(OneWire_t* OneWireStruct, uint8_t byte);

#endif /* ONEWIRE_H_ */
