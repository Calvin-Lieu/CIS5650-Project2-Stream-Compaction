#include <cstdio>
#include "cpu.h"

#include <vector>

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            odata[0] = idata[0];
        	for (int i = 1; i < n; i++)
        	{
                odata[i] = odata[i - 1] + idata[i];
        	}
            for (int i = n - 1; i > 0; i--)
            {
                odata[i] = odata[i - 1];
            }
            odata[0] = 0;
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int idx = 0;
            for (int i = 0; i < n; i++)
            {
	            if (idata[i] != 0)
	            {
                    odata[idx] = idata[i];
                    idx++;
	            }
            }
            timer().endCpuTimer();
            return idx;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int *isZero = new int[n];
            int *scanArr = new int[n];
            for (int i = 0; i < n; i++)
            {
                isZero[i] = (idata[i] != 0 ? 1 : 0);
            }
            scan(n, scanArr, isZero);
            for (int i = 0; i < n; i++)
            {
	            if (isZero[i] == 1)
	            {
                    odata[scanArr[i]] = idata[i];
	            }
            }
            timer().endCpuTimer();
            return scanArr[n - 1] + isZero[n - 1];
        }
    }
}
