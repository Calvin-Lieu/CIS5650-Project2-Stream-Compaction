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
        void scan(int n, int* odata, const int* idata) {
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

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Cuda memcpy device to host");

            timer().endGpuTimer();
        }

        __global__ void upSweepOpt(int n, int* data, int offset)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int numNodes = n / offset;
        	if (idx >= numNodes) return;

        	int right = (idx + 1) * offset - 1;
            int left = right - (offset >> 1);

            if (left < 0) return;
            data[right] += data[left];
        }

        __global__ void downSweepOpt(int n, int* data, int offset)
        {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int numNodes = n / offset;
        	if (idx >= numNodes) return;

        	int right = (idx + 1) * offset - 1;
            int left = right - (offset >> 1);

            if (left < 0) return;
            int temp = data[left];
            data[left] = data[right];
            data[right] += temp;
        
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scanOpt(int n, int *odata, const int *idata) {
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
            checkCUDAError("Cuda memset dev_in padding");

            int blockSize = 256;

            timer().startGpuTimer();
            // Up-sweep phase
            for (int i = 0; i < ilog2ceil(n); i++)
            {
                int offset = 1 << (i + 1);
                int numNodes = nPadded / offset;
                int actualBlocks = (numNodes + blockSize - 1) / blockSize;
                upSweepOpt << <actualBlocks, blockSize >> > (nPadded, dev_in, offset);
                checkCUDAError("upSweep launch");
            }

            // Set last element to zero
            cudaMemset(dev_in + (nPadded - 1), 0, sizeof(int));
            checkCUDAError("Cuda memset between up and down sweep");

            // Down-sweep phase
            for (int i = ilog2ceil(n) - 1; i >= 0; i--)
            {
                int offset = 1 << (i + 1);
                int numNodes = nPadded / offset;
                int actualBlocks = (numNodes + blockSize - 1) / blockSize;
                downSweepOpt << <actualBlocks, blockSize >> > (nPadded, dev_in, offset);
                checkCUDAError("downSweep launch");
            }
            cudaDeviceSynchronize();
            timer().endGpuTimer();

            cudaMemcpy(odata, dev_in, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Cuda memcpy device to host");

            cudaFree(dev_in);
            checkCUDAError("Cuda free dev_in");
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
            int nPadded = 1 << ilog2ceil(n);
            // Allocate global memory
            int* dev_in;
            int* dev_bools;
            int* dev_scanned;
            int* dev_out;

            cudaMalloc((void**)&dev_in, n * sizeof(int));
            checkCUDAError("Cuda malloc dev_in");
            cudaMalloc((void**)&dev_bools, nPadded * sizeof(int));
            checkCUDAError("Cuda malloc dev_bools");
            cudaMalloc((void**)&dev_scanned, nPadded * sizeof(int));
            checkCUDAError("Cuda malloc dev_scanned");
            cudaMalloc((void**)&dev_out, n * sizeof(int));
            checkCUDAError("Cuda malloc dev_out");

            int blockSize = 256;
            int blocks = (n + blockSize - 1) / blockSize;

            cudaMemcpy(dev_in, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("Cuda memcpy dev_in  host to device");

            // Pre-zero padding for exclusive scan result
            cudaMemset(dev_scanned + n, 0, (nPadded - n) * sizeof(int));
            checkCUDAError("Cuda memset dev_scanned padding");

        	timer().startGpuTimer();

            // Map input to booleans
            StreamCompaction::Common::kernMapToBoolean << <blocks, blockSize >> > (n, dev_bools, dev_in);

            // Copy bool values to dev_scanned so we can do in-place scan without losing bools
            cudaMemcpy(dev_scanned, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAError("Cuda memcpy dev_bools to dev_scanned");

            // Up-sweep phase
            for (int i = 0; i < ilog2ceil(n); i++)
            {
                int offset = 1 << (i + 1);
                int numNodes = nPadded / offset;
                int actualBlocks = (numNodes + blockSize - 1) / blockSize;
                upSweepOpt << <actualBlocks, blockSize >> > (nPadded, dev_scanned, offset);
                checkCUDAError("upSweep launch");
            }

            // Set last element to zero
            cudaMemset(dev_scanned + (nPadded - 1), 0, sizeof(int));
            checkCUDAError("Cuda memset between up and down sweep");

            // Down-sweep phase
            for (int i = ilog2ceil(n) - 1; i >= 0; i--)
            {
                int offset = 1 << (i + 1);
                int numNodes = nPadded / offset;
                int actualBlocks = (numNodes + blockSize - 1) / blockSize;
                downSweepOpt << <actualBlocks, blockSize >> > (nPadded, dev_scanned, offset);
                checkCUDAError("downSweep launch");
            }

            // Before scatter, get scan result + last boolean for return
            int lastTwo[2];
            cudaMemcpy(lastTwo, dev_scanned + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Cuda memcpy exclusive scan result");
        	cudaMemcpy(lastTwo + 1, dev_bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
        	checkCUDAError("Cuda memcpy last boolean");

            int scatterRemaining = lastTwo[0] + lastTwo[1];

            // Scatter step
            StreamCompaction::Common::kernScatter<<<blocks, blockSize>>>(n, dev_out, dev_in, dev_bools, dev_scanned);
            cudaDeviceSynchronize();
        	timer().endGpuTimer();

            // Bring results to host
            cudaMemcpy(odata, dev_out, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("Cuda memcpy to odata device to host");

            // Free memory
            cudaFree(dev_in);
            checkCUDAError("Cuda free dev_in");
            cudaFree(dev_bools);
            checkCUDAError("Cuda free dev_bools");
            cudaFree(dev_scanned);
            checkCUDAError("Cuda free dev_scanned");
            cudaFree(dev_out);
            checkCUDAError("Cuda free dev_out");

            return scatterRemaining;
        }
    }
}
