#include "onewire.h"



void ONEWIRE_OUTPUT(OneWire_t* OneWireStruct){
	// output mode
	/*OneWireStruct->GPIOx->MODER &= 0xFCFFFFFF;
	OneWireStruct->GPIOx->MODER |= OneWireStruct->GPIO_Pin;*/
	GPIOC->MODER &= 0xFCFFFFFF;
	GPIOC->MODER |= 0x01000000;
}

void ONEWIRE_INPUT(OneWire_t* OneWireStruct){
	// input mode
	//OneWireStruct->GPIOx->MODER &= 0xFCFFFFFF;
	GPIOC->MODER &= 0xFCFFFFFF;
}

void ONEWIRE_LOW(OneWire_t* OneWireStruct){
	//SET_REG(OneWireStruct->GPIOx->ODR,OneWireStruct->GPIO_Pin,0);
	SET_REG(GPIOC->ODR,0x03000000,0);
}

void ONEWIRE_DELAY_us(int sec){
	for(int i=0; i<sec; i++){
		Delay_us();
	}
}

void ONEWIRE_DELAY_10ms(int sec){
	for(int i=0; i<sec; i++){
		Delay_10ms();
	}
}

/* Init OneWire Struct with GPIO information
 * param:
 *   OneWire: struct to be initialized
 *   GPIOx: Base of the GPIO DQ used, e.g. GPIOA
 *   GPIO_Pin: The pin GPIO DQ used, e.g. 5
 */
void OneWire_Init(OneWire_t* OneWireStruct, GPIO_TypeDef* GPIOx, uint32_t GPIO_Pin) {
	// TODO
	OneWireStruct->GPIOx = GPIOx;
	OneWireStruct->GPIO_Pin = GPIO_Pin;
}

/* Send reset through OneWireStruct
 * Please implement the reset protocol
 * param:
 *   OneWireStruct: wire to send
 * retval:
 *    0 -> Reset OK
 *    1 -> Reset Failed
 */
uint8_t OneWire_Reset(OneWire_t* OneWireStruct) {
	// TODO
	//...
	/* Line low, and wait 480us */
	ONEWIRE_INPUT(OneWireStruct);
	ONEWIRE_LOW(OneWireStruct);
	ONEWIRE_OUTPUT(OneWireStruct);
	ONEWIRE_DELAY_us(480);

	/* Release line and wait for 70us */
	ONEWIRE_INPUT(OneWireStruct);
	ONEWIRE_DELAY_us(70);

	/* Check bit value */
	//int agree = ((OneWireStruct->GPIOx->IDR >> 12) & 1)^1;
	int agree = ((GPIOC->IDR >> 12) & 1)^1;

	/* Delay for 410 us */
	ONEWIRE_DELAY_us(410);

	return (uint8_t)agree;
}

/* Write 1 bit through OneWireStruct
 * Please implement the send 1-bit protocol
 * param:
 *   OneWireStruct: wire to send
 *   bit: bit to send
 */
void OneWire_WriteBit(OneWire_t* OneWireStruct, uint8_t bit) {
	// TODO
	OneWire_Reset(OneWireStruct);

	ONEWIRE_DELAY_us(5); //>1us
	ONEWIRE_INPUT(OneWireStruct);
	if (bit) {
		/* Set line low */
		ONEWIRE_LOW(OneWireStruct);
		ONEWIRE_OUTPUT(OneWireStruct);
		ONEWIRE_DELAY_us(5); //<15
		/* Bit high */
		ONEWIRE_INPUT(OneWireStruct);
		ONEWIRE_DELAY_us(10);
	}
	else {
		/* Set line low */
		ONEWIRE_LOW(OneWireStruct);
		ONEWIRE_OUTPUT(OneWireStruct);
		ONEWIRE_DELAY_us(90);
	}

	ONEWIRE_DELAY_us(5); //>1us
	ONEWIRE_INPUT(OneWireStruct);
}



/* Read 1 bit through OneWireStruct
 * Please implement the read 1-bit protocol
 * param:
 *   OneWireStruct: wire to read from
 */
uint8_t OneWire_ReadBit(OneWire_t* OneWireStruct) {
	// TODO
		OneWire_Reset(OneWireStruct);

		ONEWIRE_INPUT(OneWireStruct);

		ONEWIRE_LOW(OneWireStruct);
		ONEWIRE_OUTPUT(OneWireStruct);
		ONEWIRE_DELAY_us(5); //>1us

		ONEWIRE_INPUT(OneWireStruct);
		//uint8_t bit=((OneWireStruct->GPIOx->IDR >> 12) & 1);
		uint8_t bit=((GPIOC->IDR >> 12) & 1);

		ONEWIRE_DELAY_us(5); //>1us
		ONEWIRE_INPUT(OneWireStruct);//?

		return bit;
}

/* A convenient API to write 1 byte through OneWireStruct
 * Please use OneWire_WriteBit to implement
 * param:
 *   OneWireStruct: wire to send
 *   byte: byte to send
 */
void OneWire_WriteByte(OneWire_t* OneWireStruct, uint8_t byte) {
	// TODO
	uint8_t bit;

	for(int i=0;i<8;i++){
		bit = byte&1;
		OneWire_WriteBit(OneWireStruct,bit);
		byte = byte>>1;
	}

}
void write_three_things(OneWire_t* OneWireStruct, uint8_t byte){
	// TODO
		uint8_t bit;
		uint8_t H=1000000;
		uint8_t L=0;

		for(int i=0;i<8;i++){
			bit = H&1;
			OneWire_WriteBit(OneWireStruct,bit);
			H = H>>1;
		}
		for(int i=0;i<8;i++){
			bit = L&1;
			OneWire_WriteBit(OneWireStruct,bit);
			L = L>>1;
		}
		for(int i=0;i<8;i++){
			bit = byte&1;
			OneWire_WriteBit(OneWireStruct,bit);
			byte = byte>>1;
		}
}

/* A convenient API to read 1 byte through OneWireStruct
 * Please use OneWire_ReadBit to implement
 * param:
 *   OneWireStruct: wire to read from
 */
uint8_t OneWire_ReadByte(OneWire_t* OneWireStruct) {
	// TODO
	uint8_t temp;
	uint8_t byte=0;
	for(int i=0;i<8;i++){
		temp = OneWire_ReadBit(OneWireStruct);
		byte=byte+(temp<<i);
	}

	return byte;
}

/* Send ROM Command, Skip ROM, through OneWireStruct
 * You can use OneWire_WriteByte to implement
 */
void OneWire_SkipROM(OneWire_t* OneWireStruct) {
	// TODO
	OneWire_WriteByte(OneWireStruct,0xCC);
}
