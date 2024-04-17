#ifndef KERNELS_HPP
#define KERNELS_HPP

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_device_functions.h>
#include <algorithm>
#include <vector>
#include <array>

#define HIP_CHECK(command){ \
    hipError_t status = command; \
    if(status != hipSuccess) { \
        std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
        std::abort(); \
    } \
}

template <typename VecType, typename DimType>
__global__ void set_to_zero_debug(
    VecType* __restrict__ x,
    unsigned* __restrict__ wg_data,
    long long int* __restrict__ timer_data,
    DimType total_size)
{
    if(threadIdx.x == 0)
    {
        wg_data[blockIdx.x] = __smid();
        timer_data[blockIdx.x] = wall_clock64();
    }
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        x[idx] = 0;
    }
}

template <bool WG_REVERSAL, typename VecType, typename DimType>
__global__ void add_two_debug(
    VecType* __restrict__ x,
    unsigned* __restrict__ wg_data,
    long long int* __restrict__ timer_data,
    DimType total_size)
{
    if(threadIdx.x == 0)
    {
        wg_data[blockIdx.x] = __smid();
        timer_data[blockIdx.x] = wall_clock64();
    }
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if constexpr(WG_REVERSAL)
    {
        auto bid = gridDim.x - blockIdx.x - 1;
        idx = bid * blockDim.x + threadIdx.x;
    }
    if (idx < total_size)
    {
        x[idx] = x[idx] + 2;
    }
}
#endif
