#ifndef COMMON_HPP
#define COMMON_HPP

#include <hip/hip_runtime.h>
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
__global__ void add_one(
    VecType* __restrict__ x,
    DimType total_size)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        x[idx] = 0;
    }
}

template <bool WG_REVERSAL, typename VecType, typename DimType>
__global__ void add_two(
    VecType* __restrict__ x,
    DimType total_size)
{
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
