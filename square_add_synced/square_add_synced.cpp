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
    int in_height = 1024;
    int in_width = 1024;
    int out_channels = 1024; // AKA: number of filters
    bool use_wg_reversal = false;
};

input_params parse_inputs(int argc, char** argv)
{
    argparse::ArgumentParser program("square_add_synced");
    program.add_argument("in_height")
        .help("input height")
        .scan<'i', int>();
    program.add_argument("in_width")
        .help("input width")
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
        program.get<int>("in_height"),
        program.get<int>("in_width"),
        program.get<int>("out_channels"),
        program.get<bool>("--wg-rev")
    };
    return ip;
}

int main(int argc, char * argv[])
{
    auto ip = parse_inputs(argc, argv);

    std::cout << "input_dims: [" <<  ip.out_channels << ", " << ip.in_height << ", " << ip.in_width << "]\n";
    std::cout << "wg_reversal: " << ip.use_wg_reversal << "\n";

    std::size_t input_size = ip.out_channels * ip.in_height * ip.in_width;

    std::vector<data_type> A_vec(input_size);
    std::vector<data_type> C_vec(input_size);
    std::vector<data_type> D_vec(input_size);

    // fill A and C with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> distrib(0.0, 1.0);
    std::generate(A_vec.begin(), A_vec.end(), [&](){return distrib(gen);});
    std::generate(C_vec.begin(), C_vec.end(), [&](){return distrib(gen);});
    
    // set up device buffers
    data_type* gpu_A;
    data_type* gpu_C;
    std::size_t bytes_A = A_vec.size() * sizeof(data_type);
    std::size_t bytes_C = C_vec.size() * sizeof(data_type);
    HIP_CHECK(hipMalloc(&gpu_A, bytes_A));
    HIP_CHECK(hipMalloc(&gpu_C, bytes_C));

    // set up threads
    std::size_t block_size = 220;
    std::size_t grid_size = ip.out_channels;
    
    // blocking copies
    HIP_CHECK(hipMemcpy(gpu_A, A_vec.data(), bytes_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gpu_C, C_vec.data(), bytes_C, hipMemcpyHostToDevice));
    
    auto square_kernel = square<data_type, int>;
    hipLaunchKernelGGL(square_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        gpu_A,
        ip.in_height,
        ip.in_width,
        input_size
    );

    // trying to prefetch gpu_C results
    auto add_one_kernel = add_one<data_type, int>;
    hipLaunchKernelGGL(add_one_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        gpu_C,
        ip.in_height,
        ip.in_width,
        input_size
    );

    auto add_kernel = vector_add<false, data_type, int>;
    if(ip.use_wg_reversal)
    {
        add_kernel = vector_add<true, data_type, int>;
    }
    hipLaunchKernelGGL(add_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        gpu_A,
        gpu_C,
        ip.in_height,
        ip.in_width,
        input_size
    );

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(D_vec.data(), gpu_C, bytes_C, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(gpu_A));
    HIP_CHECK(hipFree(gpu_C));
}
