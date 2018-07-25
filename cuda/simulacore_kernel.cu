// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

#define BINARYSIZE 8368
#define NUMOFCORES 384

__device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__ unsigned char getNextByte(int *targetMem, int longInt) {
	return ((targetMem[longInt >> 2] & (0xff << ( 8 * (longInt % 4) ))) >> ( 8 * (longInt % 4) ));
}

__global__ void simulacore_gpu(int *targetMem, int *resultMem) {
	int coreNum = 0;
	int opcAddress = 0;
	int varAddress = 0;
	unsigned char command;
	// fake register
	long rbp_4, rbp_8, eax, ecx; // rsp not needed
	coreNum = getGlobalIdx_1D_1D();
	if (coreNum < NUMOFCORES) {
		opcAddress = 0xf80;
		for ( int step = 0; step < 16; step++ ) {
			command = getNextByte(targetMem,opcAddress); opcAddress++;
			if ( command == 0x55 ) {
				// push rbp # 55
			}	
			if ( command == 0x5d ) {
				// pop rbp # 5d
			}	
			if ( command == 0x48 ) {	
				// mov        rbp, rsp	# 48 89 E5 
			}
			if ( command == 0xc7 ) {	
				command = getNextByte(targetMem,opcAddress); opcAddress++;
				command = getNextByte(targetMem,opcAddress); opcAddress++;
				if ( command == 0xfc ) {
					// mov        dword [rbp+var_4], 0x0	# C7 45 FC 00 00 00 00  
					rbp_4 = (getNextByte(targetMem,opcAddress)); opcAddress++;
					rbp_4 = rbp_4 + (getNextByte(targetMem,opcAddress) << 8); opcAddress++;
					rbp_4 = rbp_4 + (getNextByte(targetMem,opcAddress) << 16); opcAddress++;
					rbp_4 = rbp_4 + (getNextByte(targetMem,opcAddress) << 24); opcAddress++;
				}
				if ( command == 0xf8 ) {
					// mov        dword [rbp+var_8], 0x7	# C7 45 F8 02 00 00 00   
					rbp_8 = (getNextByte(targetMem,opcAddress)); opcAddress++;
					rbp_8 = rbp_8 + (getNextByte(targetMem,opcAddress) << 8); opcAddress++;
					rbp_8 = rbp_8 + (getNextByte(targetMem,opcAddress) << 16); opcAddress++;
					rbp_8 = rbp_8 + (getNextByte(targetMem,opcAddress) << 24); opcAddress++;
				}
			}
			if ( command == 0x8B ) {	
				command = getNextByte(targetMem,opcAddress); opcAddress++;
				if ( command == 0x05 ) {	
					// mov        eax, dword [_a]	# 8B 05 68 00 00 00  
					varAddress = 0x1000;
					eax = (getNextByte(targetMem,varAddress)); varAddress++;
					eax = eax + (getNextByte(targetMem,varAddress) << 8); varAddress++;
					eax = eax + (getNextByte(targetMem,varAddress) << 16); varAddress++;
					eax = eax + (getNextByte(targetMem,varAddress) << 24); 
					opcAddress = opcAddress + 4;
				}
				if ( command == 0x0d ) {
					// mov        ecx, dword [_a] # 8B 0D 5B 00 00 00 
					varAddress = 0x1000;
					ecx = (getNextByte(targetMem,varAddress)); varAddress++;
					ecx = ecx + (getNextByte(targetMem,varAddress) << 8); varAddress++;
					ecx = ecx + (getNextByte(targetMem,varAddress) << 16); varAddress++;
					ecx = ecx + (getNextByte(targetMem,varAddress) << 24); 
					opcAddress = opcAddress + 4;
				}
			}
			if ( command == 0x0f ) {	
				command = getNextByte(targetMem,opcAddress); opcAddress++;
				command = getNextByte(targetMem,opcAddress); opcAddress++;
				if ( command == 0x45 ) {
					// imul       eax, dword [rbp+var_8]	# 0F AF 45 F8 
					eax = eax * rbp_8;
					command = getNextByte(targetMem,opcAddress); opcAddress++;
				}
				if ( command == 0xc1 ) {
					// imul       eax, ecx	# 0F AF C1  
					eax = eax * ecx;
					command = getNextByte(targetMem,opcAddress); opcAddress++;
				}
			}
			if ( command == 0x03 ) {
				command = getNextByte(targetMem,opcAddress); opcAddress++;
				if ( command == 0x45 ) {
					// add        eax, dword [rbp+var_8]	# 03 45 F8  
					eax = eax + rbp_8;
					command = getNextByte(targetMem,opcAddress); opcAddress++;
				}
				if ( command == 0x4d ) {
					// add        ecx, dword [rbp+var_8]	# 03 4D F8  
					ecx = ecx + rbp_8;
					command = getNextByte(targetMem,opcAddress); opcAddress++;
				}
			}
			resultMem[coreNum] = eax;
		}
	}
}
