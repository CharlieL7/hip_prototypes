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

template <typename src_data_t, typename dst_data_t>
inline __device__ __host__ dst_data_t cast_to(const src_data_t& val)
{
    return static_cast<dst_data_t>(val);
}

/**
 * Doesn't look like this needs pre-padding, but assumes symmetrical padding
 */
template <bool ASSUME_PACKED, typename src_data_t, typename acc_data_t, typename dst_data_t>
__global__ void naive_conv_fwd_nchw(
    const src_data_t* __restrict__ p_in, // input buffer
    const src_data_t* __restrict__ p_wei, // weights buffer
    dst_data_t* __restrict__ p_out, // output buffer
    const std::array<int, 5> in_strides, // strides in 5 length array, not used for ASSUME_PACKED
    const std::array<int, 5> wei_strides, // weight strides in 5 length array, not used for ASSUME_PACKED
    const std::array<int, 5> out_strides, // output strides in 5 length array, not used for ASSUME_PACKED
    int hi, // input height
    int wi, // input width
    int n, // batch size
    int k_per_group, // output channels per group
    int c_per_group, // input channels per group
    int ho, // output height
    int wo, // output width
    int sy, // kernel y stride
    int sx, // kernel x stride
    int dy, // dilation y axis
    int dx, // dilation x axis
    int py, // padding y axis
    int px, // padding x axis
    int fy, // weights y axis
    int fx, // weights x axis
    int group // number of groups
)
{
    /*
     *  need to compute total output pixel: `group * n * k_per_group * ho * wo`.
     *  to distribute this workload, let one workgroup compute `ho * wo` pixel,
     *  hence need `group * n * k_per_group` workgroups (grid_size).
     *
     *  Charlie: for a group size of 1, a given workgroup calculates the output of one of the filters
     *  over the whole input tensor.
     *  Charlie: my line comments are only for the case of group = 1, n = 1, bid = [0, k-1]
     */
    int k             = k_per_group * group;
    int c             = c_per_group * group;
    int thread_length = ho * wo;
    int bid           = blockIdx.x;
    int ik            = bid % k_per_group; // == bid
    int in            = (bid / k_per_group) % n; // == 0
    int ig            = bid / (n * k_per_group); // == 0 

    if constexpr(ASSUME_PACKED)
    {
        // p_in += 0
        p_in +=
            static_cast<size_t>(in) * c * hi * wi + static_cast<size_t>(ig) * c_per_group * hi * wi;

        // p_wei += bid * c * fy * fx
        p_wei += static_cast<size_t>(ig) * k_per_group * c_per_group * fy * fx +
                 static_cast<size_t>(ik) * c_per_group * fy * fx;
        
        // p_out += bid * ho * wo
        p_out += static_cast<size_t>(in) * k * ho * wo +
                 static_cast<size_t>(ig) * k_per_group * ho * wo +
                 static_cast<size_t>(ik) * ho * wo;
    }
    else
    {
        p_in += static_cast<size_t>(in) * in_strides[4] + static_cast<size_t>(ig) * in_strides[3];

        p_wei +=
            static_cast<size_t>(ig) * wei_strides[4] + static_cast<size_t>(ik) * wei_strides[3];

        p_out += static_cast<size_t>(in) * out_strides[4] +
                 static_cast<size_t>(ig) * out_strides[3] +
                 static_cast<size_t>(ik) * out_strides[2];
    }

    for(int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        // threads do the work of  multiples blockDim.x 
        int iho = tid / wo; // index height output
        int iwo = tid % wo; // index width output

        acc_data_t value = 0; // accumulator

        // iterate over channels
        for(int ic = 0; ic < c_per_group; ic++)
        {
            // iterate over kernel height
            for(int iy = 0; iy < fy; iy++)
            {
                // effective boolean for if a valid height index
                int valid_h = 1;
                // stride y * output_height_index - padding_y + dilation_y * kernel_height_index
                int cur_h   = sy * iho - py + dy * iy; // iho - py + iy
                if(cur_h < 0 || cur_h >= hi)
                    valid_h &= 0;
                // iterate over kernel width
                for(int ix = 0; ix < fx; ix++)
                {
                    int valid_w = 1;
                    int cur_w   = sx * iwo - px + dx * ix; // iwo - px + ix
                    if(cur_w < 0 || cur_w >= wi)
                        valid_w &= 0;

                    if(valid_w & valid_h)
                    {
                        if constexpr(ASSUME_PACKED)
                        {
                            size_t i_idx = static_cast<size_t>(ic) * hi * wi +
                                           static_cast<size_t>(cur_h) * wi +
                                           static_cast<size_t>(cur_w);

                            size_t f_idx = static_cast<size_t>(ic) * fy * fx +
                                           static_cast<size_t>(iy) * fx + static_cast<size_t>(ix);

                            value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                        }
                        else
                        {
                            size_t i_idx = static_cast<size_t>(ic) * in_strides[2] +
                                           static_cast<size_t>(cur_h) * in_strides[1] +
                                           static_cast<size_t>(cur_w) * in_strides[0];

                            size_t f_idx = static_cast<size_t>(ic) * wei_strides[2] +
                                           static_cast<size_t>(iy) * wei_strides[1] +
                                           static_cast<size_t>(ix) * wei_strides[0];

                            value += cast_to<src_data_t, acc_data_t>(p_in[i_idx]) *
                                     cast_to<src_data_t, acc_data_t>(p_wei[f_idx]);
                        }
                    }
                }
            }
        }
        if constexpr(ASSUME_PACKED)
        {
            size_t o_idx = static_cast<size_t>(iho) * wo + static_cast<size_t>(iwo);

            p_out[o_idx] = cast_to<acc_data_t, dst_data_t>(value);
        }
        else
        {
            size_t o_idx = static_cast<size_t>(iho) * out_strides[1] +
                           static_cast<size_t>(iwo) * out_strides[0];

            p_out[o_idx] = cast_to<acc_data_t, dst_data_t>(value);
        }
    }
}

template <bool WG_REVERSAL, typename VecType, typename DimType>
__global__ void vector_add(
    const VecType* __restrict__ a,
    VecType* __restrict__ b,
    DimType height,
    DimType width,
    DimType total_size) 
{
    // each workgroup calculates for one K channel
    // expecting # channels workgroups
    int thread_length = height * width;
    int bid = blockIdx.x;
    if constexpr(WG_REVERSAL)
    {
        bid = gridDim.x - blockIdx.x - 1;
    }
    int prepend_length = bid * thread_length;
    a += prepend_length;
    b += prepend_length;
    for (int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        if (tid + prepend_length < total_size)
        {
            b[tid] = a[tid] + b[tid];
        }
    }
}

template <typename VecType, typename DimType>
__global__ void add_one(
    VecType* __restrict__ x,
    DimType height,
    DimType width,
    DimType total_size)
{
    // each workgroup calculates for one K channel
    // expecting # channels workgroups
    int thread_length = height * width;
    int prepend_length = blockIdx.x * thread_length;
    x += prepend_length;
    for (int tid = threadIdx.x; tid < thread_length; tid += blockDim.x)
    {
        if (tid + prepend_length < total_size)
        {
            x[tid] = x[tid] + 1;
        }
    }
}

template <typename T, typename AccType>
std::vector<T> ref_convolution_add(
    const std::vector<T>& A,
    const std::vector<T>& W,
    const std::vector<T>& C,
    std::size_t A_height,
    std::size_t A_width,
    std::size_t out_channels,
    std::size_t in_channels,
    std::size_t kernel_height,
    std::size_t kernel_width
)
{
    std::vector<T> ret(C.size());

    // input padded with zeros for "same" output shape
    int padded_height = A_height + kernel_height - 1;
    int padded_width = A_width + kernel_width - 1;
    std::vector<float> A_padded(padded_height * padded_width * in_channels, 0.);
    for(int c = 0; c < in_channels; ++c)
    {
        for(int i = 0; i < A_height; ++i)
        {
            for(int j = 0; j < A_width; ++j)
            {
                A_padded.at(
                    c * padded_height * padded_width +
                    (i + static_cast<int>(kernel_height / 2)) * padded_width +
                    (j + static_cast<int>(kernel_width / 2))
                ) = A.at(
                    c * A_height * A_width + i * A_width + j
                );
            }
        }
    }
    
    for(int k = 0; k < out_channels; ++k)
    {
        for(int y = 0; y < A_height; ++y)
        {
            for(int x = 0; x < A_width; ++x)
            {
                AccType sum = 0.0;
                for(int c = 0; c < in_channels; ++c)
                {
                    for(unsigned int m = 0; m < kernel_height; ++m)
                    {
                        for(unsigned int n = 0; n < kernel_width; ++n)
                        {
                            unsigned int mask_index =
                                k * in_channels * kernel_height * kernel_width + 
                                c * kernel_height * kernel_width +
                                m * kernel_width +
                                n;
                            unsigned int input_index = c * padded_height * padded_width +
                                (y + m) * padded_width +
                                (x + n);
                            sum += A_padded.at(input_index) * W.at(mask_index);
                        } // n
                    } // m
                } // c
                ret[(k * A_height * A_width + y * A_width + x)] = static_cast<T>(sum);
            } // x
        } // y
    } // k

    std::transform(C.begin(), C.end(), ret.begin(), ret.begin(),
        [](auto c, auto d){ return c + d + 1; });
    return ret;
}
#endif
