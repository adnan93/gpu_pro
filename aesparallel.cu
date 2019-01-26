//Kartik Kulkarni and Eugene Linkov
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define Nb 4			// number of columns in the state & expanded key
#define Nk 4			// number of columns in a key
#define Nr 10			// number of rounds in encryption
//const int MaxThreadsPerBlock = _THREADS_PER_BLOCK;
const int MaxThreadsPerBlock = 16;
typedef unsigned char uchar;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// The global data
__device__ __constant__ uchar _Sbox[256];
__device__ __constant__ uchar _InvSbox[256];
__device__ __constant__ uchar _Xtime2Sbox[256];
__device__ __constant__ uchar _Xtime3Sbox[256];
__device__ __constant__ uchar _Xtime2[256];
__device__ __constant__ uchar _Xtime9[256];
__device__ __constant__ uchar _XtimeB[256];
__device__ __constant__ uchar _XtimeD[256];
__device__ __constant__ uchar _XtimeE[256];


extern uchar Sbox[256];
extern uchar InvSbox[256];
extern uchar Xtime2Sbox[256];
extern uchar Xtime3Sbox[256];
extern uchar Xtime2[256];
extern uchar Xtime9[256];
extern uchar XtimeB[256];
extern uchar XtimeD[256];
extern uchar XtimeE[256];

void _ShiftRows (uchar *state);
void _MixSubColumns (uchar *state);
void _AddRoundKey (unsigned *state, unsigned *key);
void _Encrypt (uchar *in, uchar *expkey, uchar *out);
extern void ExpandKey (uchar *key, uchar *expkey);



double gettime()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

//extern "C" void gpu_init(int argc, char** argv);
//extern "C" double gpu_encrypt_string(char* in, int in_len, char* key, char* out);
//extern "C" void ExpandKey(uchar *key, uchar *expkey);

__device__ void _ShiftRows (uchar *state)
{
	//printf(" ShiftRows entered\n");
	uchar tmp;
	
	// substitute row 0
	state[0] = _Sbox[state[0]], state[4]  = _Sbox[state[4]];
	state[8] = _Sbox[state[8]], state[12] = _Sbox[state[12]];

	// rotate the row 1
	tmp = _Sbox[state[1]], state[1] = _Sbox[state[5]];
	state[5] = _Sbox[state[9]], state[9] = _Sbox[state[13]], state[13] = tmp;

	// rotate the row 2
	tmp = _Sbox[state[2]], state[2] = _Sbox[state[10]], state[10] = tmp;
	tmp = _Sbox[state[6]], state[6] = _Sbox[state[14]], state[14] = tmp;

	// rotate the row 3
	tmp = _Sbox[state[15]], state[15] = _Sbox[state[11]];
	state[11] = _Sbox[state[7]], state[7] = _Sbox[state[3]], state[3] = tmp;
}

__device__ void _InvMixSubColumns (uchar *state)
{
	//printf("  - InvMixSubColumns\n");
	uchar tmp[4 * Nb];
	int i;

	// restore column 0
	tmp[0] = _XtimeE[state[0]] ^ _XtimeB[state[1]] ^ _XtimeD[state[2]] ^ _Xtime9[state[3]];
	tmp[5] = _Xtime9[state[0]] ^ _XtimeE[state[1]] ^ _XtimeB[state[2]] ^ _XtimeD[state[3]];
	tmp[10] = _XtimeD[state[0]] ^ _Xtime9[state[1]] ^ _XtimeE[state[2]] ^ _XtimeB[state[3]];
	tmp[15] = _XtimeB[state[0]] ^ _XtimeD[state[1]] ^ _Xtime9[state[2]] ^ _XtimeE[state[3]];

	// restore column 1
	tmp[4] = _XtimeE[state[4]] ^ _XtimeB[state[5]] ^ _XtimeD[state[6]] ^ _Xtime9[state[7]];
	tmp[9] = _Xtime9[state[4]] ^ _XtimeE[state[5]] ^ _XtimeB[state[6]] ^ _XtimeD[state[7]];
	tmp[14] = _XtimeD[state[4]] ^ _Xtime9[state[5]] ^ _XtimeE[state[6]] ^ _XtimeB[state[7]];
	tmp[3] = _XtimeB[state[4]] ^ _XtimeD[state[5]] ^ _Xtime9[state[6]] ^ _XtimeE[state[7]];

	// restore column 2
	tmp[8] = _XtimeE[state[8]] ^ _XtimeB[state[9]] ^ _XtimeD[state[10]] ^ _Xtime9[state[11]];
	tmp[13] = _Xtime9[state[8]] ^ _XtimeE[state[9]] ^ _XtimeB[state[10]] ^ _XtimeD[state[11]];
	tmp[2]  = _XtimeD[state[8]] ^ _Xtime9[state[9]] ^ _XtimeE[state[10]] ^ _XtimeB[state[11]];
	tmp[7]  = _XtimeB[state[8]] ^ _XtimeD[state[9]] ^ _Xtime9[state[10]] ^ _XtimeE[state[11]];

	// restore column 3
	tmp[12] = _XtimeE[state[12]] ^ _XtimeB[state[13]] ^ _XtimeD[state[14]] ^ _Xtime9[state[15]];
	tmp[1] = _Xtime9[state[12]] ^ _XtimeE[state[13]] ^ _XtimeB[state[14]] ^ _XtimeD[state[15]];
	tmp[6] = _XtimeD[state[12]] ^ _Xtime9[state[13]] ^ _XtimeE[state[14]] ^ _XtimeB[state[15]];
	tmp[11] = _XtimeB[state[12]] ^ _XtimeD[state[13]] ^ _Xtime9[state[14]] ^ _XtimeE[state[15]];

	for( i=0; i < 4 * Nb; i++ )
		state[i] = _InvSbox[ tmp[i] ];
}

__device__ void _InvShiftRows (uchar *state)
{
	//printf("  - InvShiftRows\n");
	uchar tmp;

	// restore row 0
	state[0] = _InvSbox[state[0]], state[4] = _InvSbox[state[4]];
	state[8] = _InvSbox[state[8]], state[12] = _InvSbox[state[12]];

	// restore row 1
	tmp = _InvSbox[state[13]], state[13] = _InvSbox[state[9]];
	state[9] = _InvSbox[state[5]], state[5] = _InvSbox[state[1]], state[1] = tmp;

	// restore row 2
	tmp = _InvSbox[state[2]], state[2] = _InvSbox[state[10]], state[10] = tmp;
	tmp = _InvSbox[state[6]], state[6] = _InvSbox[state[14]], state[14] = tmp;

	// restore row 3
	tmp = _InvSbox[state[3]], state[3] = _InvSbox[state[7]];
	state[7] = _InvSbox[state[11]], state[11] = _InvSbox[state[15]], state[15] = tmp;
}



__device__ void _MixSubColumns (uchar *state)
{
	//printf(" MixSubColumns entered\n");
	uchar tmp[4 * Nb];

	// mixing the column 0
	tmp[0] = _Xtime2Sbox[state[0]] ^ _Xtime3Sbox[state[5]] ^ _Sbox[state[10]] ^ _Sbox[state[15]];
	tmp[1] = _Sbox[state[0]] ^ _Xtime2Sbox[state[5]] ^ _Xtime3Sbox[state[10]] ^ _Sbox[state[15]];
	tmp[2] = _Sbox[state[0]] ^ _Sbox[state[5]] ^ _Xtime2Sbox[state[10]] ^ _Xtime3Sbox[state[15]];
	tmp[3] = _Xtime3Sbox[state[0]] ^ _Sbox[state[5]] ^ _Sbox[state[10]] ^ _Xtime2Sbox[state[15]];

	// mixing th column 1
	tmp[4] = _Xtime2Sbox[state[4]] ^ _Xtime3Sbox[state[9]] ^ _Sbox[state[14]] ^ _Sbox[state[3]];
	tmp[5] = _Sbox[state[4]] ^ _Xtime2Sbox[state[9]] ^ _Xtime3Sbox[state[14]] ^ _Sbox[state[3]];
	tmp[6] = _Sbox[state[4]] ^ _Sbox[state[9]] ^ _Xtime2Sbox[state[14]] ^ _Xtime3Sbox[state[3]];
	tmp[7] = _Xtime3Sbox[state[4]] ^ _Sbox[state[9]] ^ _Sbox[state[14]] ^ _Xtime2Sbox[state[3]];

	// mixing the column 2
	tmp[8] = _Xtime2Sbox[state[8]] ^ _Xtime3Sbox[state[13]] ^ _Sbox[state[2]] ^ _Sbox[state[7]];
	tmp[9] = _Sbox[state[8]] ^ _Xtime2Sbox[state[13]] ^ _Xtime3Sbox[state[2]] ^ _Sbox[state[7]];
	tmp[10]  = _Sbox[state[8]] ^ _Sbox[state[13]] ^ _Xtime2Sbox[state[2]] ^ _Xtime3Sbox[state[7]];
	tmp[11]  = _Xtime3Sbox[state[8]] ^ _Sbox[state[13]] ^ _Sbox[state[2]] ^ _Xtime2Sbox[state[7]];

	// mixing the column 3
	tmp[12] = _Xtime2Sbox[state[12]] ^ _Xtime3Sbox[state[1]] ^ _Sbox[state[6]] ^ _Sbox[state[11]];
	tmp[13] = _Sbox[state[12]] ^ _Xtime2Sbox[state[1]] ^ _Xtime3Sbox[state[6]] ^ _Sbox[state[11]];
	tmp[14] = _Sbox[state[12]] ^ _Sbox[state[1]] ^ _Xtime2Sbox[state[6]] ^ _Xtime3Sbox[state[11]];
	tmp[15] = _Xtime3Sbox[state[12]] ^ _Sbox[state[1]] ^ _Sbox[state[6]] ^ _Xtime2Sbox[state[11]];

	for ( int i = 0; i < 4*Nb; ++i )
		state[i] = tmp[i];
}


__device__ void _AddRoundKey (unsigned *state, unsigned *key)
{
	//printf(" AddRoundKey entered\n");
	for ( int i = 0; i < 4; ++i )
		state[i] ^= key[i];
}



// Decrypt one block
__global__ void _Decrypt (uchar *in, uchar *expkey, uchar *out)
{
	//printf("- Decrypt\n");
	//printf("  - In:  "); dump(16, in); printf("\n");
	

	__shared__ uchar buffer[16*MaxThreadsPerBlock];

	// Constant data about this thread
	const int block_start = blockIdx.x * blockDim.x * 16;
	const int thread_offset = threadIdx.x * 16;
	
	//printf ("\n cuda decrypt: blockId is %d blockDim is %d thread id is %d block_start = %d thread_offset = %d",blockIdx.x,blockDim.x,threadIdx.x , block_start, thread_offset  );

	// Read in the message data for the whole block
	for ( int i = 0; i < 16; ++i )
	{
		unsigned int index = blockDim.x * i + threadIdx.x;
		buffer[index] = in[block_start + index];
	}
	__syncthreads();

	_AddRoundKey( (unsigned*)(buffer+thread_offset), (unsigned*)expkey + Nr * Nb );
	_InvShiftRows( buffer+thread_offset );
	for ( int round = Nr; round--; )
	{
		_AddRoundKey( (unsigned*)(buffer+thread_offset), (unsigned*)expkey + round * Nb );
		if ( round )
			_InvMixSubColumns( buffer+thread_offset );
	}
	__syncthreads();

	// Copy back the memory
	for ( int i = 0; i < 16; ++i )
	{
		unsigned int index = blockDim.x * i + threadIdx.x;
		out[block_start + index] = buffer[index];
	}
}

__global__ void _Encrypt (uchar *in, uchar *expkey, uchar *out)
{
	//printf(" Encrypt entered\n");

	__shared__ uchar buffer[16*MaxThreadsPerBlock];

	// thread info
	const int block_start = blockIdx.x * blockDim.x * 16;
	const int thread_offset = threadIdx.x * 16;

	// get the stuff from gobal memory and cahche it in shared
	for ( int i = 0; i < 16; ++i )
	{
		unsigned int index = blockDim.x * i + threadIdx.x;
		buffer[index] = in[block_start + index];
	}
	//if (threadIdx.x == 0)
	//{
	//	printf ("\nGPU: input carried form global to shared:\n");
	//	int idx = 0;
     	//	for( idx = 0; idx < 16; idx++ )
     	//		//printf ("%.2x ", buffer[idx]);

		//printf ("\n");
		
	//}
	__syncthreads();

	_AddRoundKey( (unsigned*)(buffer+thread_offset), (unsigned*)expkey );


	for( int round = 1; round < Nr + 1; round++ )
	{

		//__syncthreads();

		//if (threadIdx.x == 0)
		//{
		//	printf ("\nGPU: after AddRoundKey - round %d:\n", round);
		//	int idx = 0;
	     	//	for( idx = 0; idx < 16; idx++ )
	     	//		printf ("%.2x ", buffer[idx]);
		//
		//	printf ("\n");
		
		//}	
		//__syncthreads();

		if( round < Nr )
			_MixSubColumns( buffer+thread_offset );
		else
			_ShiftRows( buffer+thread_offset );

		//_AddRoundKey( (unsigned*)(buffer+thread_offset), (unsigned*)expkey + round * Nb );
		_AddRoundKey( (unsigned*)(buffer+thread_offset), (unsigned*)expkey + round * Nb);


		
	}
	__syncthreads();

	// dunp back to the global
	for ( int i = 0; i < 16; ++i )
	{
		unsigned int index = blockDim.x * i + threadIdx.x;
		out[block_start + index] = buffer[index];
	}
}


__host__ double gpu_decrypt_string(uchar* in, int length, uchar* key, uchar* out, uchar * expkey)
{
	
	double elapsed = 0.0;
	
	//uchar expkey[4 * Nb * (Nr + 1)];
	//ExpandKey( (uchar*)key, (uchar*)expkey );

	// Allocate memory for the GPU
	uchar* d_in = 0;
	uchar* d_out = 0;
	uchar* d_expkey = 0;
	cudaMalloc( (void **) &d_in, length ) ;
	cudaMalloc( (void **) &d_out, length ) ;
	cudaMalloc( (void **) &d_expkey, 4 * Nb * (Nr + 1) );

	

	// Copy memory to the GPU
	cudaMemcpy( (void*)d_in, (void*)in, length, cudaMemcpyHostToDevice ) ;
	cudaMemcpy( (void*)d_expkey, (void*)expkey, 4 * Nb * (Nr + 1), cudaMemcpyHostToDevice ) ;
	
	uint units = length / 16;
	uint threads = (units >= MaxThreadsPerBlock) ? MaxThreadsPerBlock : (units % MaxThreadsPerBlock);
	uint blocks = units/threads;

	//unsigned int timer;
	//cutCreateTimer(&timer);
	cudaThreadSynchronize();
	//cutStartTimer(timer);

	//printf("Debug: Invoking kernel _Decrypt <<<%u,%u>>> (d_in, d_expkey, d_out)\n", blocks, threads);
	_Decrypt <<<blocks, threads>>> (d_in, d_expkey, d_out);
	//_Decrypt <<<1, 16>>> (d_in, d_expkey, d_out);

	//CUT_CHECK_ERROR("Error in kernel.");
	//printf("Success!\n");

	// Stop the timer
	cudaThreadSynchronize();
	//cutStopTimer(timer) ;
	//elapsed = cutGetTimerValue(timer);
	//cutDeleteTimer(timer);
	
	

	// Copy the results back to the CPU
        cudaMemcpy( (void*)out, (void*)d_out, length, cudaMemcpyDeviceToHost );

	cudaFree(d_in) ;
	cudaFree(d_out) ;
	return elapsed / 1000.0;
}


__host__ double gpu_encrypt_string(uchar* in, int length, uchar* key,  uchar* out, uchar * expkey)
{

	//printf ("\nGPU: Samplein:\n");
	//int idx = 0;
       	//for( idx = 0; idx < 16; idx++ )
        //     printf ("%.2x ", in[idx]);
	
	double elapsed = 0.0;
	//uchar expkey[4 * Nb * (Nr + 1)];
	//ExpandKey(reinterpret_cast<uchar *>( key), expkey );

	// GPU memory allocation
	uchar* d_in = 0;
	uchar* d_out = 0;
	uchar* d_expkey = 0;
	cudaMalloc( (void **) &d_in, length ) ;
	cudaMalloc( (void **) &d_out, length ) ;
	cudaMalloc( (void **) &d_expkey, 4 * Nb * (Nr + 1) ) ;

	// Ccopy stuff to GPU
	cudaMemcpy( (void*)d_in, (void*)in, length, cudaMemcpyHostToDevice ) ;
	cudaMemcpy( (void*)d_expkey, (void*)expkey, 4 * Nb * (Nr + 1), cudaMemcpyHostToDevice ) ;
	uint units = length / 16;
	uint threads = (units >= MaxThreadsPerBlock) ? MaxThreadsPerBlock : (units % MaxThreadsPerBlock);
	uint blocks = units/threads;

	//unsigned int timer;
	//CUT_SAFE_CALL( cutCreateTimer(&timer) );
	cudaThreadSynchronize();
	//CUT_SAFE_CALL( cutStartTimer(timer) );


	double start = gettime();

	//printf("trying kernel _Encrypt <<<%u,%u>>> (d_in, d_expkey, d_out)\n", blocks, threads);
	//printf ("\n launching cuda kernel:  blocks: %d threads: %d ", blocks, threads );
	_Encrypt <<<blocks, threads>>> (d_in, d_expkey, d_out);
	
	//_Encrypt <<<1, 1>>> (d_in, d_expkey, d_out);

	//CUT_CHECK_ERROR("Error in kernel.");
	//printf("I think no error\n");

	
	// Stop the timer
	cudaThreadSynchronize();
	//CUT_SAFE_CALL( cutStopTimer(timer) );
	//elapsed = cutGetTimerValue(timer);
	//CUT_SAFE_CALL( cutDeleteTimer(timer) );

	double end = gettime();

	//printf ("\n from device code: total encryption time is %f", end - start);


	// copy stuff back to GPU
	cudaMemcpy( (void*)out, (void*)d_out, length, cudaMemcpyDeviceToHost ) ;

	cudaFree(d_in) ;
	cudaFree(d_out) ;

	return elapsed/1000.0;
}



//__host__ void gpu_init(int argc, char** argv)
__host__ void gpu_init()
{
	printf( "starting and init cuda\n" ); 
	//CUT_DEVICE_INIT(argc, argv);
	//CUT_DEVICE_INIT ();
	printf( "initialization happened\n" );

	printf( "copying stuff to constant and global memory" );
	cudaMemcpyToSymbol( "_Sbox",       Sbox,       256 ) ;
	cudaMemcpyToSymbol( "_InvSbox",    InvSbox,    256 ) ;
	 cudaMemcpyToSymbol( "_Xtime2Sbox", Xtime2Sbox, 256 ) ;
	 cudaMemcpyToSymbol( "_Xtime3Sbox", Xtime3Sbox, 256 ) ;
	cudaMemcpyToSymbol( "_Xtime2",     Xtime2,     256 ) ;
	 cudaMemcpyToSymbol( "_Xtime9",     Xtime9,     256 ) ;
	 cudaMemcpyToSymbol( "_XtimeB",     XtimeB,     256 ) ;
	cudaMemcpyToSymbol( "_XtimeD",     XtimeD,     256 ) ;
	 cudaMemcpyToSymbol( "_XtimeE",     XtimeE,     256 ) ;
	cudaThreadSynchronize();
	printf( "copied stuff to constant and global memoryn\n" );
}















