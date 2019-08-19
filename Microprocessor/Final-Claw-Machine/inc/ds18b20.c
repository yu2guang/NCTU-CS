#include "ds18b20.h"

/* Send ConvT through OneWire with resolution
 * param:
 *   OneWire: send through this
 *   resolution: temperature resolution
 * retval:
 *    0 -> OK
 *    1 -> Error
 */
int DS18B20_ConvT(OneWire_t* OneWire, DS18B20_Resolution_t resolution) {
	// TODO

	// reset
	OneWire_Reset(OneWire);

	// ROM
	OneWire_SkipROM(OneWire);

	//write command
	OneWire_WriteByte(OneWire,0x4E);//write command

	write_three_things(OneWire,0x44);
	// command: write scratchpad
	//OneWire_WriteByte(OneWire,0x44);

	// delay 375ms
	ONEWIRE_DELAY_10ms(38);

	return 0;
}

/* Read temperature from OneWire
 * param:
 *   OneWire: send through this
 *   destination: output temperature
 * retval:
 *    0 -> OK
 *    1 -> Error
 */
uint8_t DS18B20_Read(OneWire_t* OneWire, float *destination) {
	// TODO

	// reset
	OneWire_Reset(OneWire);

	// ROM
	OneWire_SkipROM(OneWire);
	//read

	OneWire_WriteByte(OneWire,0xBE);//read command
	// command: read scratchpad
	uint8_t LS = OneWire_ReadByte(OneWire);
	uint8_t MS = OneWire_ReadByte(OneWire);

	int one_bit;
	int mult = -3;
	int two;
	float temp = 0;
	for(int i=1; i<=7; i++){//LS
		two = 1;
		one_bit = (LS>>i)&1;
		if(mult<0){
			for(int j=0; j<(mult*(-1)); j++){
				two *= 2;
			}
			temp += one_bit/two;
		}
		else if(mult==1){
			temp += one_bit;
		}
		else if(mult>0){
			for(int j=0; j<mult; j++){
				two *= 2;
			}
			temp += one_bit*two;
		}

		mult++;
	}

	mult = 4;
	for(int i=0; i<3; i++){//MS
		two = 1;
		one_bit = (MS>>i)&1;
		for(int j=0; j<mult; j++){
			two *= 2;
		}
					temp += one_bit*two;

	}

	destination =&temp;

	return 0;
}

/* Set resolution of the DS18B20
 * param:
 *   OneWire: send through this
 *   resolution: set to this resolution
 * retval:
 *    0 -> OK
 *    1 -> Error
 */
uint8_t DS18B20_SetResolution(OneWire_t* OneWire, DS18B20_Resolution_t resolution) {
	// TODO

	// reset
	OneWire_Reset(OneWire);

	// ROM
	OneWire_SkipROM(OneWire);

	// command: write scratchpad
	OneWire_WriteByte(OneWire,0x4E);

	// data: resolution(0.125), 11-bit

	write_three_things(OneWire,0b01011111);

	return 0;
}

/* Check if the temperature conversion is done or not
 * param:
 *   OneWire: send through this
 * retval:
 *    0 -> OK
 *    1 -> Not yet
 */
uint8_t DS18B20_Done(OneWire_t* OneWire) {
	// TODO
	return 0;
}
