#include <stdio.h>
#include <stdlib.h>
#include "simulacore_kernel.cu"

int main(void) {
	int col, row;
	uint8_t *ptr_executable;
	uint8_t *ptr_executableCopy;
	int *ptr_result;
	FILE *ptr_fp;
    int *d_binary;
    int *d_result;
	ptr_executable = (uint8_t *)malloc(BINARYSIZE);
	ptr_executableCopy = (uint8_t *)malloc(BINARYSIZE * NUMOFCORES);
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

	if(fread(ptr_executable, BINARYSIZE * sizeof( uint8_t ), 1, ptr_fp) != 1)
	{
		printf( "Read error!\n" );
		exit( 1 );
	}
	fclose(ptr_fp);

	// cp the executable #cores times
	for ( int repeatCp = 0; repeatCp < NUMOFCORES; repeatCp++ ) {
		memcpy(ptr_executableCopy + (repeatCp * BINARYSIZE), ptr_executable, BINARYSIZE);
	}

	for ( int i = 0; i < (NUMOFCORES); i++ ) {
		ptr_result[i] = 0x44;
	}

	for (row = 0x00; row < 0x10f; row++ ) {
		printf("%00002x: ", (row * 0x10));
		for(col = 0; col < 0x10; col++)
			printf("%02x ", ptr_executableCopy[(row * 0x10) + col]);
		printf("\n");
	}
	// exit ( 1 );

	dim3 blocksPerGrid(1,1,1); //use only one block
	dim3 threadsPerBlock(NUMOFCORES,1,1); //use N threads in the block myKernel<<<blocksPerGrid, threadsPerBlock>>>(result);
    
    checkCudaErrors(cudaMalloc((int**)&d_binary, (NUMOFCORES * BINARYSIZE)));
    checkCudaErrors(cudaMemcpy(d_binary, ptr_executableCopy, (NUMOFCORES * BINARYSIZE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMalloc((int**)&d_result, NUMOFCORES * sizeof( int )));
    checkCudaErrors(cudaMemcpy(d_result, ptr_result, NUMOFCORES * sizeof( int ), cudaMemcpyHostToDevice));
 	// start the i86 opcode interpreter on the GPU   
    simulacore_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_binary, d_result);
    
    checkCudaErrors(cudaMemcpy(ptr_executableCopy, d_binary, (NUMOFCORES * BINARYSIZE), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(ptr_result, d_result, NUMOFCORES * sizeof( int ), cudaMemcpyDeviceToHost));

	printf("The numbers read from GPU memory are:\n");
	for (row = 0x00; row < 0x10f; row++ ) {
		printf("%00002x: ", (row * 0x10));
		for(col = 0; col < 0x10; col++)
			printf("%02x ", ptr_executableCopy[(row * 0x10) + col + (2 * BINARYSIZE)]);
		printf("\n");
	}

	printf("\n");

	for ( int i = 0; i < NUMOFCORES; i++ ) {
		printf("%d:\t%02x \n", i, ptr_result[i]);
	}
	printf("\n");

	free(ptr_executable);
	free(ptr_executableCopy);
	return 0;
}
