#include <random>
#include <iostream>
#include <hip/hip_runtime.h>
#include "common.hpp"

/**
 * Calculation:
 * A = input tensor
 * W = kernel tensor
 * B = conv(A, W)
 * C = other constant
 * D = add(B, C)
 */
int main(int argc, char * argv[])
{
    using data_type = float;
    // input buffer sizes
    std::size_t batch_size = 1;
    std::size_t in_channels = 1024;
    std::size_t in_width = 1024;
    std::size_t in_height = 1024;

    // kernel buffer sizes
    std::size_t kernel_width = 3;
    std::size_t kernel_height = 3;
    std::size_t out_channels = 1024; // AKA: number of filters

    std::cout << "A_dims: [" << batch_size << ", " << in_channels << ", " << in_height << ", " << in_width << "]\n";
    std::cout << "W_dims: [" << out_channels << ", " << in_channels << ", " << kernel_height << ", " << kernel_width << "]\n";
    std::cout << "C_dims: [" << batch_size << ", " << out_channels << ", " << in_height << ", " << in_width << "]\n";

    std::size_t conv_input_size = batch_size * in_channels * in_height * in_width;
    std::size_t conv_output_size = batch_size * out_channels * in_height * in_width;
    std::size_t kernel_size = out_channels * in_channels *  kernel_height * kernel_width;

    std::vector<data_type> A_vec(conv_input_size);
    std::vector<data_type> W_vec(kernel_size);
    std::vector<data_type> B_vec(conv_output_size);
    std::vector<data_type> C_vec(conv_output_size);
    std::vector<data_type> D_vec(conv_output_size);

    // fill A, W, and C with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);
    std::generate(A_vec.begin(), A_vec.end(), [&](){return distrib(gen);});
    std::generate(W_vec.begin(), W_vec.end(), [&](){return distrib(gen);});
    std::generate(C_vec.begin(), C_vec.end(), [&](){return distrib(gen);});
    
    // Debug: fill with 1's and 0's
    //std::generate(A_vec.begin(), A_vec.end(), [&](){return 1.0;});
    //std::generate(W_vec.begin(), W_vec.end(), [&](){return 1.0;});
    //std::generate(C_vec.begin(), C_vec.end(), [&](){return 0.0;});

    // make C pinned memory
    data_type* C_pinned_data;
    HIP_CHECK(hipHostMalloc(&C_pinned_data, C_vec.size() * sizeof(data_type)));
    std::copy(C_vec.begin(), C_vec.end(), C_pinned_data);

    // set up device buffers
    data_type* gpu_A;
    data_type* gpu_W;
    data_type* gpu_B;
    data_type* gpu_C;
    data_type* gpu_D;
    std::size_t bytes_A = A_vec.size() * sizeof(data_type);
    std::size_t bytes_W = W_vec.size() * sizeof(data_type);
    std::size_t bytes_B = B_vec.size() * sizeof(data_type);
    std::size_t bytes_C = C_vec.size() * sizeof(data_type);
    std::size_t bytes_D = D_vec.size() * sizeof(data_type);
    HIP_CHECK(hipMalloc(&gpu_A, bytes_A));
    HIP_CHECK(hipMalloc(&gpu_W, bytes_W));
    HIP_CHECK(hipMalloc(&gpu_B, bytes_B));
    HIP_CHECK(hipMalloc(&gpu_C, bytes_C));
    HIP_CHECK(hipMalloc(&gpu_D, bytes_D));

    // set up threads
    std::size_t block_size = 220;
    std::size_t conv_grid_size = out_channels;

    // set up streams
    hipStream_t compute_stream;
    hipStream_t memory_stream;
    HIP_CHECK(hipStreamCreate(&compute_stream));
    HIP_CHECK(hipStreamCreate(&memory_stream));
    
    // blocking copies
    HIP_CHECK(hipMemcpy(gpu_A, A_vec.data(), bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu_W, W_vec.data(), bytes_W, hipMemcpyHostToDevice));
    
    auto conv_kernel = naive_conv_fwd_nchw<true, data_type, data_type, data_type>;
    hipLaunchKernelGGL(conv_kernel,
        dim3(conv_grid_size),
        dim3(block_size),
        0,
        0,
        gpu_A,
        gpu_W,
        gpu_B,
        {},
        {},
        {},
        in_height,
        in_width,
        batch_size,
        out_channels,
        in_channels,
        in_height,
        in_width,
        1, // kernel y stride
        1, // kernel x stride
        1, // dilation y
        1, // dilation x
        1, // padding y
        1, // padding x
        kernel_height,
        kernel_width,
        1 // groups
    );

    HIP_CHECK(hipMemcpyAsync(gpu_C, C_pinned_data, bytes_C, hipMemcpyHostToDevice, memory_stream));
    HIP_CHECK(hipStreamSynchronize(memory_stream));
    auto add_kernel = vector_add<false, data_type, int>;
    std::size_t add_grid_size = (conv_output_size + block_size - 1) / block_size;
    hipLaunchKernelGGL(add_kernel,
        dim3(add_grid_size),
        dim3(block_size),
        0,
        0,
        gpu_B,
        gpu_C,
        gpu_D,
        conv_output_size
    );

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(D_vec.data(), gpu_D, bytes_D, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(gpu_A));
    HIP_CHECK(hipFree(gpu_B));
    HIP_CHECK(hipFree(gpu_C));
    HIP_CHECK(hipFree(gpu_D));
    HIP_CHECK(hipFree(gpu_W));
    HIP_CHECK(hipHostFree(C_pinned_data));

    HIP_CHECK(hipStreamDestroy(compute_stream));
    HIP_CHECK(hipStreamDestroy(memory_stream));
	
    // Debug: ref version
    //auto ref_vals = ref_convolution_add<data_type, data_type>(A_vec, W_vec, C_vec, in_height, in_width, out_channels, in_channels, kernel_width, kernel_height);
    //
    //if(
    //    std::equal(
    //        ref_vals.begin(),
    //        ref_vals.end(),
    //        D_vec.begin(),
    //        [](auto ref, auto x){ return std::abs(ref - x) < 1e-3; }
    //    )
    //)
    //{
    //    std::cout<<"Passed!\n";
    //}
    //else
    //{
    //    std::cout<<"Failed!\n";
    //}

    // Debug: print vectors
    //std::cout << "ref: [ ";
    //for(auto x : ref_vals)
    //{
    //    std::cout << x << ", ";
    //}
    //std::cout << "]\n";
    //std::cout << "gpu: [ ";
    //for(auto x : D_vec)
    //{
    //    std::cout << x << ", ";
    //}
    //std::cout << "]\n";

}
