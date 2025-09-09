#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void scanStep(int n, int* odata, const int* idata, int offset) {
            int idx = threadIdx.x + blockIdx.x * blockDim.x;
            if (idx >= n) return;
            if (idx >= offset)
            {
                odata[idx] = idata[idx - offset] + idata[idx];
            } else
            {
                odata[idx] = idata[idx];
            }
            
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Declare and allocate global memory
            int *dev_in, *dev_out;
            cudaMalloc((void**)&dev_in, n * sizeof(int));
            cudaMalloc((void**)&dev_out, n * sizeof(int));

            // Copy data to device
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 128;
            int blocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            for (int i = 0; i < ilog2ceil(n); i++)
            {
                int offset = 1 << (i - 1);
	            scanStep<<<blocks, blockSize>>>(n)
            }
            timer().endGpuTimer();

            // Copy result back to CPU
            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            // Free global memory
            cudaFree(dev_in);
            cudaFree(dev_out);
        }
    }
}
