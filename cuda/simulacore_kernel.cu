// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include <helper_functions.h>
#include <helper_cuda.h>

/* TODO:                                *
 * Replace with dynamic linked PTX-JIT  *
 * see: CUDA/samples/ptxjit             */

__device__ int getGlobalIdx_1D_1D()
{
	return blockIdx.x *blockDim.x + threadIdx.x;
}

__device__ int getDWReverse(int coreNum, int *arch, uint8_t *targetMem, int executableOffset) {
	int returnValue = 0;
	int varAddress = 0;
	if (arch[coreNum] == MACHO_I86 ) 
		varAddress = executableOffset + 0x1000; 
	if (arch[coreNum] == LIELF_I86 ) 
		varAddress = executableOffset + 0x1030; 
	returnValue = targetMem[varAddress]; varAddress++;
	returnValue = returnValue + (targetMem[varAddress] << 8); varAddress++;
	returnValue = returnValue + (targetMem[varAddress] << 16); varAddress++;
	returnValue = returnValue + (targetMem[varAddress] << 24); 
	return returnValue;
}
__global__ void simulacore_gpu(int *arch, uint8_t *targetMem, int *resultMem) {
	int coreNum = 0;
	int executableOffset = 0;
	int opcAddress = 0;
	int varAddress = 0;
	unsigned char command;
	// fake register
	long rbp_4, rbp_8, eax, ecx, edx; // rsp not needed
	coreNum = getGlobalIdx_1D_1D();
	if (coreNum < NUMOFCORES) {
		// opcAddress = 0xf80; // MachO
		if (arch[coreNum] == MACHO_I86 ) {
			if (arch[coreNum] != arch[0] ) {
				executableOffset = (NUM_OF_MACHO_CORES * BINARYSIZE_LIELF) + ((coreNum - NUM_OF_MACHO_CORES) * BINARYSIZE_LIELF);
			} else {
				executableOffset = (coreNum * BINARYSIZE_LIELF);
			}
			opcAddress = executableOffset + 0xf80;
		}
		if (arch[coreNum] == LIELF_I86 ) {
			if (arch[coreNum] != arch[0] ) {
				executableOffset = (NUM_OF_MACHO_CORES * BINARYSIZE_LIELF) + ((coreNum - NUM_OF_MACHO_CORES) * BINARYSIZE_LIELF);
			} else {
				executableOffset = (coreNum * BINARYSIZE_LIELF);
			}
			opcAddress = executableOffset + 0x4d6;
		}
		for ( int step = 0; step < 16; step++ ) {
			command = targetMem[opcAddress]; opcAddress++;
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
				command = targetMem[opcAddress]; opcAddress++;
				command = targetMem[opcAddress]; opcAddress++;
				if ( command == 0xfc ) {
					// mov        dword [rbp+var_4], 0x0	# C7 45 FC 00 00 00 00  
					rbp_4 = targetMem[opcAddress]; opcAddress++;
					rbp_4 = rbp_4 + (targetMem[opcAddress] << 8); opcAddress++;
					rbp_4 = rbp_4 + (targetMem[opcAddress] << 16); opcAddress++;
					rbp_4 = rbp_4 + (targetMem[opcAddress] << 24); opcAddress++;
				}
				if ( command == 0xf8 ) {
					// mov        dword [rbp+var_8], 0x7	# C7 45 F8 02 00 00 00   
					rbp_8 = (targetMem[opcAddress]); opcAddress++;
					rbp_8 = rbp_8 + (targetMem[opcAddress] << 8); opcAddress++;
					rbp_8 = rbp_8 + (targetMem[opcAddress] << 16); opcAddress++;
					rbp_8 = rbp_8 + (targetMem[opcAddress] << 24); opcAddress++;
				}
			}
			if ( command == 0x8B ) {	
				command = targetMem[opcAddress]; opcAddress++;
				if ( command == 0x05 ) {	
					// mov        eax, dword [_a]	# 8B 05 68 00 00 00 
					eax = getDWReverse(coreNum, arch, targetMem, executableOffset);
					opcAddress = opcAddress + 4;
				}
				if ( command == 0x0d ) {
					// mov        ecx, dword [_a] # 8B 0D 5B 00 00 00 
					ecx = getDWReverse(coreNum, arch, targetMem, executableOffset);
					opcAddress = opcAddress + 4;
				}
				if ( command == 0x55 ) {
					//  mov        edx, dword [rbp+var_4] # 8B 55 FC 
					edx = rbp_4;
					opcAddress++; 
				}
				if ( command == 0x45 ) {
					//  mov        eax, dword [rbp+var_4] # 8B 45 FC 
					eax = rbp_4;
					opcAddress++; 
				}
			}
			if ( command == 0x0f ) {	
				opcAddress++;
				command = targetMem[opcAddress]; opcAddress++;
				if ( command == 0x45 ) {
					// imul       eax, dword [rbp+var_8]	# 0F AF 45 F8 
					command = targetMem[opcAddress]; opcAddress++;
					if ( command == 0xf8 ) // OSX RBP address downwards from ff 
						eax = eax * rbp_8;
					if ( command == 0xfc ) // Linux ELF
						eax = eax * rbp_4;
				}
				if ( command == 0xc1 ) {
					// imul       eax, ecx	# 0F AF C1  
					eax = eax * ecx;
				}
				if ( command == 0xd0 ) {
					// imul       edx, eax	# 0F AF D0  
					edx = edx * eax;
				}
			}
			if ( command == 0x03 ) {
				command = targetMem[opcAddress]; opcAddress++;
				if ( command == 0x45 ) {
					// add        eax, dword [rbp+var_8]	# 03 45 F8  
					eax = eax + rbp_8;
					command = targetMem[opcAddress]; opcAddress++;
				}
				if ( command == 0x4d ) {
					// add        ecx, dword [rbp+var_8]	# 03 4D F8  
					ecx = ecx + rbp_8;
					command = targetMem[opcAddress]; opcAddress++;
				}
			}
			if ( command == 0x83 ) {
				// add        eax, 0x1 # 83 C0 01 
				eax = eax + 0x01; // Linux gcc -O0 ???
				opcAddress++; opcAddress++;
			}
			if ( command == 0x01 ) {
				command = targetMem[opcAddress]; opcAddress++;
				if ( command == 0xd0 ) {
					// add        eax, edx # 01 D0 
					eax = eax + edx;
				}
				if ( command == 0xca ) {
					// add        edx, ecx # 01 CA 
					edx = edx + ecx; 
				}
			}
			resultMem[coreNum] = eax; // result is stored in register eax and transfered to resultMem
		}
	}
}
