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
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_size)
    {
        x[idx] = 0;
    }
    unsigned se_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_SE_ID_SIZE-1, HW_ID_SE_ID_OFFSET, HW_ID));
    unsigned xcc_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(XCC_ID_XCC_ID_SIZE - 1, XCC_ID_XCC_ID_OFFSET, XCC_ID));
    unsigned cu_id = __builtin_amdgcn_s_getreg(
            GETREG_IMMED(HW_ID_CU_ID_SIZE - 1, HW_ID_CU_ID_OFFSET, HW_ID));
    printf("workgroup number (blockIdx.x): %u, ", blockIdx.x);
    printf("se_id : %u, ", se_id);
    printf("xcc_id : %u, ", xcc_id);
    printf("cu_id : %u\n", cu_id);
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
