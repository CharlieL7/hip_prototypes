#include <random>
#include <iostream>
#include <hip/hip_runtime.h>
#include "conv_add.hpp"

#define HIP_CHECK(command){ \
    hipError_t status = command; \
    if(status != hipSuccess) { \
        std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
        std::abort(); \
    } \
}

int main(int argc, char * argv[])
{
    // doing add(conv(A, W), C)
    
    // set up buffer sizes
    std::size_t in_width = 2048;
    std::size_t in_height = 2048;
    // current code is probably limited to a square kernel
    std::size_t kernel_width = 3;
    std::size_t kernel_height = 3;
    std::size_t input_size = in_width * in_height;
    std::size_t kernel_size = kernel_width * kernel_height;

    std::vector<float> W_vec = 
    {
        1.0f,  2.0f,  3.0f,
        6.0f,  5.0f,  4.0f,
        3.0f,  2.0f,  1.0f
    };

    
    // set up host buffers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 99);
    std::vector<unsigned int> A_vec(in_width * in_height);
    std::generate(A_vec.begin(), A_vec.end(), [&](){return distrib(gen);});
    int kernel_radius = kernel_width / 2;
    int padded_height = in_height + kernel_height - 1;
    int padded_width = in_width + kernel_width - 1;
    // kernel padded so after convolution it's the same shape
    std::vector<unsigned int> padded_input(padded_height * padded_width, 0);
    for(int i = 0; i < in_height; i++)
        for(int j = 0; j < in_width; j++) {
            padded_input[(i + kernel_radius) * padded_width + (j + kernel_radius)] = A_vec[i * in_width + j];
        }
    A_vec = padded_input;

    std::vector<int> B_vec(input_size);
    std::vector<int> C_vec(input_size);
    std::generate(C_vec.begin(), C_vec.end(), [&](){return distrib(gen);});
    std::vector<int> D_vec(input_size);

    // set up device buffers
    unsigned int* gpu_A;
    int* gpu_B;
    int* gpu_C;
    int* gpu_D;
    float* gpu_W;
    std::size_t bytes_A = A_vec.size() * sizeof(unsigned int);
    std::size_t bytes_B = B_vec.size() * sizeof(int);
    std::size_t bytes_C = C_vec.size() * sizeof(int);
    std::size_t bytes_D = D_vec.size() * sizeof(int);
    std::size_t bytes_W = W_vec.size() * sizeof(float);
    HIP_CHECK(hipMalloc(&gpu_A, bytes_A));
    HIP_CHECK(hipMalloc(&gpu_B, bytes_B));
    HIP_CHECK(hipMalloc(&gpu_C, bytes_C));
    HIP_CHECK(hipMalloc(&gpu_D, bytes_D));
    HIP_CHECK(hipMalloc(&gpu_W, bytes_W));
    uint2 inputDimensions = make_uint2(in_width, in_height);
    uint2 maskDimensions  = make_uint2(kernel_width, kernel_height);

    // set up threads
    std::size_t block_size = 512;
    std::size_t conv_grid_size = (input_size + block_size - 1) / block_size;
    
    // blocking copies
    HIP_CHECK(hipMemcpy(gpu_C, C_vec.data(), bytes_C, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu_A, A_vec.data(), bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu_W, W_vec.data(), bytes_W, hipMemcpyHostToDevice));
    
    hipLaunchKernelGGL(simpleNonSeparableConvolution,
        dim3(conv_grid_size),
        dim3(block_size),
        0,
        0,
        gpu_A,
        gpu_W,
        gpu_B,
        inputDimensions,
        maskDimensions,
        padded_width
    );

    
    std::size_t add_grid_size = (input_size + block_size - 1) / block_size;
    hipLaunchKernelGGL(vector_add,
        dim3(add_grid_size),
        dim3(block_size),
        0,
        0,
        gpu_D,
        gpu_B,
        gpu_C,
        in_width,
        in_height
    );

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(D_vec.data(), gpu_D, bytes_D, hipMemcpyDeviceToHost));

    auto ref_vals = ref_calc(A_vec, W_vec, C_vec, in_height, in_width, padded_width, kernel_width);
    
    if(ref_vals == D_vec)
    {
        std::cout<<"Passed!\n" << std::endl;
    }
    else
    {
        std::cout<<"Failed\n" << std::endl;
    }

    HIP_CHECK(hipFree(gpu_A));
    HIP_CHECK(hipFree(gpu_B));
    HIP_CHECK(hipFree(gpu_C));
    HIP_CHECK(hipFree(gpu_D));
    HIP_CHECK(hipFree(gpu_W));
}
