#ifndef COMMON_HPP
#define COMMON_HPP

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
__global__ void set_to_zero(
    VecType* __restrict__ x,
    DimType total_size)
{
    if(threadIdx.x == 0)
    {
            auto wall_time = wall_clock64();
            unsigned smid = __smid();
            unsigned xcc_id = smid >> 6;
            unsigned se_id = smid & 0x30 >> 4;
            unsigned cu_id = smid & 0xf;
            printf("%lli, %u, %u, %u, %u\n", wall_time, blockIdx.x, xcc_id, se_id, cu_id);
    }
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        x[idx] = 0;
    }
}

template <typename VecType, typename DimType>
__global__ void add_one(
    VecType* __restrict__ x,
    DimType total_size)
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        x[idx] = x[idx] + 1;
    }
}

template <bool WG_REVERSAL, typename VecType, typename DimType>
__global__ void add_two(
    VecType* __restrict__ x,
    DimType total_size)
{
    if(threadIdx.x == 0)
    {
            auto wall_time = wall_clock64();
            unsigned smid = __smid();
            unsigned xcc_id = smid >> 6;
            unsigned se_id = smid & 0x30 >> 4;
            unsigned cu_id = smid & 0xf;
            printf("%lli, %u, %u, %u, %u\n", wall_time, blockIdx.x, xcc_id, se_id, cu_id);
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
