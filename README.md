# Simulacore

A multicore opcode interpreter and runtime environment in CUDA.

## Run the example

```
git clone https://github.com/OpenDGPS/simulacore
cd simulacore/cuda
nvcc -I <PATH_TO_CUDA_SDK/samples/common/inc -o simulacore simulacore.cu
./simulacore
```

## What for?

Simulacore is a proof-of-concept to answer a simple question: Is a GPU able to interpret the opcode of a CISC-processor and run/interpret/simulate such a program? 

And if so, is it possible to run many of similar or even different programs in parallel? And what if the code is compiled for different processors and different operating systems? What about sharing data and scheduling tasks?

## Background

GPUs like NVIDIAs Keppler are SIMD-processors. Every core at these GPUs get the same code and after the 'GO' every core starts to execute exactly the same code. In contrast, on a classic multicore CPU every core executes its own code. Both types of processing units have the same problem: how to access the data memory without to interference with the other cores, trying to access the same memory address. On GPUs this problem is easy to solve, mostly. When and if all the cores working on the same code it is predictable that every core needs the same amount of clock cycles. The memory accessable to one core can be calculated by its core number (or thread number). To transfer data from one core to another you can sync the cores and rotate the addresses. On multicore CPUs this is much harder, at least because the os kernel handel the threads and the clock cycles depends an the specific code running on each core. There is no sync between the cores except the OS-kernel managed this. On GPUs there are features like syncthreads to stop cores until they are aligned.

## The very basic idea

Executables are a set of bytes interpreted by the progamm loader of an operating system, and shaped for the CPU disgnated to execute this specific program. Either MachO (on macOS) or ELF (on Linux) or even DOS-programs compiled for a CPU contains opcode which is interpretable for the current porcessor. Arround this code there are areas which contains data, text, system call informations and so on. 

Assume, an executable i86-binary for macOS is written to the GPU memory it should be possible to determine, which part is the data, the text and what is the opcode. To know this, it is necessary to know the executable format of the runtime environment of the compilation target. That part is easy.

Next part to know is the processor architecture or ISA of the compilation target. This is the hard part. Even RISC processors are now not very "reduced". On CISC processors an instruction with all necessary informations can be between one and more than 10 bytes long. 

If an opcode interpreter runs in a GPU core it have to go step by step through the bytes. Which means 256 different bytes for the first order opcodes and - depends on the command - interpreting additional informations and flags about source and target registers, memory addresses or alignments.

Additionally to the information about the target processing code (ISA) and the operating system it is necessary to know the endianness of the target CPU. 

## Interpreting the binary from simple.c

![Disassembly of simple.c](/disassembly.png)

The very basic MachO binary compiled from the "simple.c" via gcc (-O0 to prevent any optimizations) to "simple" is our starting point. It starts with the instructions at address 0xf80. First command is a push at register RBP. The opcode is 0x55. Next the RSP will be moved to RBP register. Then the registers RBP, EAX and ECX will be used for the calculation of the target value 42.

It is helpful to know, that the opcodes are byte oriented even if the processor (i86_64) is a little endian CPU. The codes are not stored at 32 or 64 bit memory address in reverse order. Instead the CPU reads the codes byte by byte. So, if there is a unambiguous interpretable instruction for a byte, no further commands will be readed (except to fill the instruction pipeline). 

In the sample C code there is a global integer variable "a" and a integer variable "b" inside the main-function. The compiler stores "a" in the data section at the binary and "b" is hardcoded to use in the RBP-register. Because of the RBP register is 128 bit, "b" only use a part (starting from byte number 8) of the register.

Because of the return value of the main function you need to run the "simple" executable via terminal and get the return value by "echo $?". 

### Reading the binaries from executables and transfer it to the GPU memory


In simulacore.cu the executable file "simple"(MachO) and "simple-linux-elf"(Linux Elf) is opened and copyed to the memory. 

From line 99 on the CUDA part is configured. The minimal number of threads are initialized (384 for the "GeForce GT 650M" on a "MacBook Pro (Retina, Mid 2012)". After this a memory section from the GPU device memory is allocated. Via cudaMemcpy the executable binaries are transfered to the device memory. 


### Run the i86 opcode interpreter kernel on the GPU

At line 109 in simulacore.cu the GPU kernel is called via 
```
simulacore_gpu<<<blocksPerGrid, threadsPerBlock>>>(d_arch, d_binary, d_result);
```
If it's successfully execution the result will be transfered back to the host memory and the result will be printed to confirm the correct result. Correct lines should look like this:

```
result for GPU core #97 (MachO format):	2a  
result for GPU core #98 (MachO format):	2a  
result for GPU core #99 (MachO format):	2a  
result for GPU core #100 (MachO format):	2a  
result for GPU core #101 (MachO format):	2a  
result for GPU core #102 (MachO format):	2a
...
result for GPU core #136 (Linux ELF format):	2a  
result for GPU core #137 (Linux ELF format):	2a  
result for GPU core #138 (Linux ELF format):	2a  
result for GPU core #139 (Linux ELF format):	2a  
result for GPU core #140 (Linux ELF format):	2a
```
This means cores from number 97 to 99 interpreted the executable from the MachO file ("simple") and got the correct result 0x2a (42). The cores number 136 to 140 interpreted the executable from "simple-linux-elf" as Linux Elf format and interpreted it correct to the result 0x2a (42).

The interpreter itself is located on the simulacore_kernel.cu. The function simulacore_gpu gets a pointer of the device memory for architecture configuration, executable memory and result array.

To help to understand the if-conditions, the disassembly (from Hopper Disassembler for OSX) are listed as comments. The order of the if-statements is not the exact order of the opcode in the binary.  

Even if CUDA - and in more general, GPUs - offer registers to it's cores, in this proof-of-concept the i86 registers are defined as variables. The defined C-variables are stored via MOV (0xc7) at the register variable rbp_8 and eax and the calculation happens at eax and ecx. The final result of the calculation can be found at eax. The value of eax will be written to the device memory via "resultMem[coreNum] = eax;" at line 107 in file simulacore_kernel.cu.

## Conclusion

It is shown, that it is possible to interprete and run opcode from an Intel processor with a GPU. In the history of computer this is not the first time. The Digital FX!32 had done this on Digital Alpha Workstations in the 90the. More sophisticated than this PoC the software also made runtime analytics to optimize the performance on the flight (http://www.hpl.hp.com/hpjournal/dtj/vol9num1/vol9num1art1.pdf). 

Additional it is shown, that the same executable opcode can be interpreted on many cores in parallel.

With the ability to run the same code many times in parallel (up to 4000 cores on a NVIDIA 1080ti), this solution could be faster as the target processor even if the opcodes interpreted and a GPU usually runs on lower clock than a typical Intel CPU.

Additionally the option to run different executable formats, independent from the host operating system offers new ways to build emulators of ancient computer systems like NEXTSTEP or RISC OS the legendary operating system from Acorn.

## Future prospects

Given the prerequisite, that it is possible to find a valid solution to call static or dynamic loaded system functions from the CUDA interpreter, a productive environment to use simulacore should be possible. It would mean, that a OS-kernel is possible which manages the memory and I/O transfer between the host and the device, instantiates and handles the threads, and coordinates the interprocess communication between the GPU threads. This would offer more than 50.000 threads running on an NVIDIA 1080ti, only limited by the amount of memory on the GPU. Even a Java, JavaScript or any other virtual machine could be run on the GPU.

## Next steps

~~- run the same executable many times~~
~~- run the same C code compiled for different OS in parallel~~
- run the same C code compiled for different CPUs and OS in parallel
- evaluate opcode interpretation of embedded systems like Arduino 
- evaluate timing and sync behaviour
- run benchmarks
- optimize the interpreter code by using NVIDIA PTX instructions (espacially by using byte reversal)
- generalize the interpreter code by abstracting the "X86 Opcode and Instruction Reference" XML repository
- evaluate different ways to call system functions

## Disclaimer

I am not affiliated with NVIDIA. I like CUDA and try to simulate complex systems with it. But I'm a catastrophic programmer and rarely stick to any code conventions.
