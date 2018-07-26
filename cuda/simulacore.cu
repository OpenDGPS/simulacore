#include <stdio.h>
#include <stdlib.h>

#define BINARYSIZE_MACHO 8368
#define BINARYSIZE_LIELF 8576 // 8368
#define NUMOFCORES 384

#define NUM_OF_MACHO_CORES 133
#define NUM_OF_LIELF_CORES (NUMOFCORES - NUM_OF_MACHO_CORES)

#define TOTALBINARYSIZE (BINARYSIZE_LIELF * NUMOFCORES)
#include "simulacore_kernel.cu"

int main(void) {
	uint8_t *ptr_executable;
	uint8_t *ptr_executableCopy;
	int *ptr_arch;
	int *ptr_result;
	FILE *ptr_fp;
    int *d_binary;
    int *d_arch;
    int *d_result;
	ptr_executable = (uint8_t *)malloc(BINARYSIZE_LIELF);
	ptr_executableCopy = (uint8_t *)malloc(TOTALBINARYSIZE);
	ptr_arch = (int *)malloc(NUMOFCORES * sizeof( int ));
	ptr_result = (int *)malloc(NUMOFCORES * sizeof( int ));
	if ( !ptr_executable ) {
		printf("Memory allocation error!\n");
		exit(1);
	} 
	if((ptr_fp = fopen("../simple", "rb"))==NULL)
	{
		printf("Unable to open the file!\n");
		exit(1);
	}

	if(fread(ptr_executable, BINARYSIZE_MACHO * sizeof( uint8_t ), 1, ptr_fp) != 1)
	{
		printf( "Read error!\n" );
		exit( 1 );
	}

	for (int fillIdx = BINARYSIZE_MACHO; fillIdx < BINARYSIZE_LIELF; fillIdx++) {
		ptr_executable[fillIdx] = 0xff; // smaller executable filled with 0
	}

	fclose(ptr_fp);

	for ( int repeatCp = 0; repeatCp < NUM_OF_MACHO_CORES; repeatCp++ ) {
		memcpy(ptr_executableCopy + (repeatCp * BINARYSIZE_LIELF), ptr_executable, BINARYSIZE_LIELF);
	}

	free(ptr_executable);

	ptr_executable = (uint8_t *)malloc(BINARYSIZE_LIELF);

	if((ptr_fp = fopen("../simple-linux-elf", "rb"))==NULL)
	{
		printf("Unable to open the file!\n");
		exit(1);
	}

	if(fread(ptr_executable, BINARYSIZE_LIELF * sizeof( uint8_t ), 1, ptr_fp) != 1)
	{
		printf( "Read error!\n" );
		exit( 1 );
	}

	fclose(ptr_fp);

	// cp the executable #cores times
	for ( int repeatCp = NUM_OF_MACHO_CORES; repeatCp < NUMOFCORES; repeatCp++ ) {
		memcpy(ptr_executableCopy + (NUM_OF_MACHO_CORES * BINARYSIZE_LIELF) + ((repeatCp - NUM_OF_MACHO_CORES) * BINARYSIZE_LIELF), ptr_executable, BINARYSIZE_LIELF);
	}

	for ( int i = 0; i < (NUMOFCORES); i++ ) {
		if ( i < NUM_OF_MACHO_CORES )
			ptr_arch[i] = 0x01; // MachO
		else
			ptr_arch[i] = 0x02; // Linux
	}
	for ( int i = 0; i < (NUMOFCORES); i++ ) {
		ptr_result[i] = 0x44;
	}
	// printf("%lu\n", sizeof(*ptr_executable));
	// exit ( 1 );
	/*
	int row, col;
	printf("%02x ", TOTALBINARYSIZE);

	for (row = 0; row < (TOTALBINARYSIZE >> 4); row++ ) {
		printf("%00002x: ", (row * 0x10));
		for(col = 0; col < 0x10; col++)
			printf("%02x ", ptr_executableCopy[(row * 0x10) + col]);
		printf("\n");
	}
	*/

	dim3 blocksPerGrid(1,1,1); //use only one block
	dim3 threadsPerBlock(NUMOFCORES,1,1); //use N threads in the block myKernel<<<blocksPerGrid, threadsPerBlock>>>(result);
    
    checkCudaErrors(cudaMalloc((int**)&d_binary, (TOTALBINARYSIZE)));
    checkCudaErrors(cudaMemcpy(d_binary, ptr_executableCopy, (TOTALBINARYSIZE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((int**)&d_arch, NUMOFCORES * sizeof( int )));
    checkCudaErrors(cudaMemcpy(d_arch, ptr_arch, NUMOFCORES * sizeof( int ), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((int**)&d_result, NUMOFCORES * sizeof( int )));
    checkCudaErrors(cudaMemcpy(d_result, ptr_result, NUMOFCORES * sizeof( int ), cudaMemcpyHostToDevice));
 	// start the i86 opcode interpreter on the GPU   
    simulacore_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_arch, d_binary, d_result);
    
    // checkCudaErrors(cudaMemcpy(ptr_executableCopy, d_binary, (TOTALBINARYSIZE), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(ptr_result, d_result, NUMOFCORES * sizeof( int ), cudaMemcpyDeviceToHost));


	for ( int i = 0; i < NUMOFCORES; i++ ) {
		printf("result for GPU core #%d (%s):\t%02x  \n", i, (i < NUM_OF_MACHO_CORES ? "MachO format" : "Linux ELF format"), ptr_result[i]);
	}
	printf("\n");

	
	free(ptr_executableCopy);
	return 0;
}
