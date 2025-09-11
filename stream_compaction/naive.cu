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
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
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
            checkCUDAError("Cuda Malloc");
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            checkCUDAError("Cuda Malloc");

            // Copy data to device
            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Cuda Memcpy to device");

            int blockSize = 128;
            int blocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            for (int i = 0; i < ilog2ceil(n); i++)
            {
                int offset = 1 << i;
                scanStep << <blocks, blockSize >> > (n, dev_out, dev_in, offset);
                checkCUDAError("Scan step kernel");
                std::swap(dev_in, dev_out);
            }
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            // Copy result back to CPU
            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_in, (n - 1)* sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Cuda Memcpy to host");

            // Free global memory
            cudaFree(dev_in);
            cudaFree(dev_out);
        }
    }
}
