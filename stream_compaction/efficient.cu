#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweep(int n, int* data, int offset)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            if (idx % offset == 0)
            {
                data[idx + offset - 1] += data[idx + (offset >> 1) - 1];
            }
        }

        __global__ void downSweep(int n, int* data, int offset)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx >= n) return;
            if (idx % offset == 0)
            {
                int temp = data[idx + (offset >> 1) - 1];
                data[idx + (offset >> 1) - 1] = data[idx + offset - 1];
                data[idx + offset - 1] += temp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // Calculate necessary padding for non-powers of 2, if power of two then == 0
            int nPadded = 1 << ilog2ceil(n);

            // Declare and allocate padded buffer
            int* dev_in;
            cudaMalloc((void**)&dev_in, nPadded * sizeof(int));
            checkCUDAError("Cuda Malloc");

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Cuda memcpy host to device");

            // Set padded memory to zeroes
            cudaMemset(dev_in + n, 0, (nPadded - n) * sizeof(int));

            int blockSize = 128;
            int blocks = (nPadded + blockSize - 1) / blockSize;

            timer().startGpuTimer();
            // Up-sweep phase
            for (int i = 0; i < ilog2ceil(n); i++)
            {
                int offset = 1 << (i + 1);
                upSweep << <blocks, blockSize >> > (nPadded, dev_in, offset);
            }

            // Set last element to zero
            cudaMemset(dev_in + (nPadded - 1), 0, sizeof(int));
            checkCUDAError("Cuda memset between up and down sweep");

            // Down-sweep phase
            for (int i = ilog2ceil(n) - 1; i >= 0; i--)
            {
                int offset = 1 << (i + 1);
                downSweep << <blocks, blockSize >> > (nPadded, dev_in, offset);
            }

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Cuda memcpy device to host");
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            // Allocate global memory
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
