#include <random>
#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>
#include "argparse.hpp"
#include "common.hpp"

using data_type = float;

struct input_params
{
	int batch_size = 1;
    int in_channels = 1024;
    int in_height = 1024;
    int in_width = 1024;
    int kernel_height = 3;
    int kernel_width = 3;
    int out_channels = 1024; // AKA: number of filters
    bool use_wg_reversal = false;
};

input_params parse_inputs(int argc, char** argv)
{
    argparse::ArgumentParser program("conv_add_trivial_kernel");
    program.add_argument("in_channels")
        .help("number of channels")
        .scan<'i', int>();
    program.add_argument("in_height")
        .help("input height")
        .scan<'i', int>();
    program.add_argument("in_width")
        .help("input width")
        .scan<'i', int>();
    program.add_argument("kernel_height")
        .help("kernel height")
        .scan<'i', int>();
    program.add_argument("kernel_width")
        .help("kernel width")
        .scan<'i', int>();
    program.add_argument("out_channels")
        .help("out channels")
        .scan<'i', int>();
    program.add_argument("--wg-rev")
        .help("use workgroup reversal for add(B, C)")
        .flag();
    try{
        program.parse_args(argc, argv);
    }
    catch(const std::exception& err){
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(EXIT_FAILURE);
    }
    input_params ip{
        1,
        program.get<int>("in_channels"),
        program.get<int>("in_height"),
        program.get<int>("in_width"),
        program.get<int>("kernel_height"),
        program.get<int>("kernel_width"),
        program.get<int>("out_channels"),
        program.get<bool>("--wg-rev")
    };
    return ip;
}

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
    auto ip = parse_inputs(argc, argv);

    std::cout << "A_dims: [" << ip.batch_size << ", " << ip.in_channels << ", " << ip.in_height << ", " << ip.in_width << "]\n";
    std::cout << "W_dims: [" << ip.out_channels << ", " << ip.in_channels << ", " << ip.kernel_height << ", " << ip.kernel_width << "]\n";
    std::cout << "C_dims: [" << ip.batch_size << ", " << ip.out_channels << ", " << ip.in_height << ", " << ip.in_width << "]\n";
    std::cout << "wg_reversal: " << ip.use_wg_reversal << "\n";

    std::size_t conv_input_size = ip.batch_size * ip.in_channels * ip.in_height * ip.in_width;
    std::size_t conv_output_size = ip.batch_size * ip.out_channels * ip.in_height * ip.in_width;
    std::size_t kernel_size = ip.out_channels * ip.in_channels *  ip.kernel_height * ip.kernel_width;

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

    // set up device buffers
    data_type* gpu_A;
    data_type* gpu_W;
    data_type* gpu_B;
    data_type* gpu_C;
    std::size_t bytes_A = A_vec.size() * sizeof(data_type);
    std::size_t bytes_W = W_vec.size() * sizeof(data_type);
    std::size_t bytes_B = B_vec.size() * sizeof(data_type);
    std::size_t bytes_C = C_vec.size() * sizeof(data_type);
    std::size_t bytes_D = D_vec.size() * sizeof(data_type);
    HIP_CHECK(hipMalloc(&gpu_A, bytes_A));
    HIP_CHECK(hipMalloc(&gpu_W, bytes_W));
    HIP_CHECK(hipMalloc(&gpu_B, bytes_B));
    HIP_CHECK(hipMalloc(&gpu_C, bytes_C));

    // set up threads
    std::size_t block_size = 220;
    std::size_t conv_grid_size = ip.out_channels;

    // blocking copies
    HIP_CHECK(hipMemcpy(gpu_A, A_vec.data(), bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu_W, W_vec.data(), bytes_W, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu_C, C_vec.data(), bytes_C, hipMemcpyHostToDevice));
    
    auto conv_kernel = naive_conv_prefetch_fwd_nchw<true, data_type, data_type, data_type>;
    hipLaunchKernelGGL(conv_kernel,
        dim3(conv_grid_size),
        dim3(block_size),
        0,
        0,
        gpu_A,
        gpu_W,
        gpu_C,
        gpu_B,
        {},
        {},
        {},
        ip.in_height,
        ip.in_width,
        ip.batch_size,
        ip.out_channels,
        ip.in_channels,
        ip.in_height,
        ip.in_width,
        1, // kernel y stride
        1, // kernel x stride
        1, // dilation y
        1, // dilation x
        1, // padding y
        1, // padding x
        ip.kernel_height,
        ip.kernel_width,
        1 // groups
    );

    auto add_kernel = vector_add<false, data_type, int>;
    if(ip.use_wg_reversal)
    {
        add_kernel = vector_add<true, data_type, int>;
    }
    hipLaunchKernelGGL(add_kernel,
        dim3(conv_grid_size),
        dim3(block_size),
        0,
        0,
        gpu_B,
        gpu_C,
        ip.in_height,
        ip.in_width,
        conv_output_size
    );

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(C_vec.data(), gpu_C, bytes_C, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(gpu_A));
    HIP_CHECK(hipFree(gpu_B));
    HIP_CHECK(hipFree(gpu_C));
    HIP_CHECK(hipFree(gpu_W));
	
    // Debug: ref version
    /*
    auto ref_vals = ref_convolution_add<data_type, data_type>(A_vec, W_vec, C_vec, ip.in_height, ip.in_width, ip.out_channels, ip.in_channels, ip.kernel_width, ip.kernel_height);
    
    if(
        std::equal(
            ref_vals.begin(),
            ref_vals.end(),
            D_vec.begin(),
            [](auto ref, auto x){ return std::abs(ref - x) < 1e-3; }
        )
    )
    {
        std::cout<<"Passed!\n";
    }
    else
    {
        std::cout<<"Failed!\n";
    }
    */

    // Debug: print vectors
    /*
    std::cout << "ref: [ ";
    for(auto x : ref_vals)
    {
        std::cout << x << ", ";
    }
    std::cout << "]\n";
    std::cout << "gpu: [ ";
    for(auto x : D_vec)
    {
        std::cout << x << ", ";
    }
    std::cout << "]\n";
    */

}
