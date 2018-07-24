#include <stdio.h>
#include <stdlib.h>
#include "simulacore_kernel.cu"

int main(void) {
	int col, row;
	uint8_t *ptr_d;
	FILE *ptr_fp;
    int *d_binary;
	ptr_d = (uint8_t *)malloc(BINARYSIZE);
	if ( !ptr_d ) {
		printf("Memory allocation error!\n");
		exit(1);
	} 
	if((ptr_fp = fopen("../simple", "rb"))==NULL)
	{
		printf("Unable to open the file!\n");
		exit(1);
	}

	if(fread(ptr_d, BINARYSIZE * sizeof( uint8_t ), 1, ptr_fp) != 1)
	{
		printf( "Read error!\n" );
		exit( 1 );
	}
	fclose(ptr_fp);

	for (row = 0x00; row < 0x10f; row++ ) {
		printf("%00002x: ", (row * 0x10));
		for(col = 0; col < 0x10; col++)
			printf("%02x ", ptr_d[(row * 0x10) + col]);
		printf("\n");
	}

	dim3 blocksPerGrid(1,1,1); //use only one block
	dim3 threadsPerBlock(NUMOFCORES,1,1); //use N threads in the block myKernel<<<blocksPerGrid, threadsPerBlock>>>(result);
    
    checkCudaErrors(cudaMalloc((int**)&d_binary, BINARYSIZE));
    checkCudaErrors(cudaMemcpy(d_binary, ptr_d, BINARYSIZE, cudaMemcpyHostToDevice));

 	// start the i86 opcode interpreter on the GPU   
    simulacore_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_binary);
    
    checkCudaErrors(cudaMemcpy(ptr_d, d_binary, BINARYSIZE, cudaMemcpyDeviceToHost));

	printf("The numbers read from GPU memory are:\n");
	for (row = 0x00; row < 0x10f; row++ ) {
		printf("%00002x: ", (row * 0x10));
		for(col = 0; col < 0x10; col++)
			printf("%02x ", ptr_d[(row * 0x10) + col]);
		printf("\n");
	}

	printf("\n");
	free(ptr_d);
	return 0;
}
