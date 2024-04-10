#include <random>
#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>
#include "argparse.hpp"
#include "common.hpp"

using data_type = float;

struct input_params
{
	int N = 100;
    bool use_wg_reversal = false;
};

input_params parse_inputs(int argc, char** argv)
{
    argparse::ArgumentParser program("add_one_add_two");
    program.add_argument("N")
        .help("size of vector to add to")
        .scan<'i', int>();
    program.add_argument("--wg-rev")
        .help("use workgroup reversal for add_two")
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
        program.get<int>("N"),
        program.get<bool>("--wg-rev")
    };
    return ip;
}

int main(int argc, char * argv[])
{
    auto ip = parse_inputs(argc, argv);

    std::cout << "N: " << ip.N << "\n";
    std::cout << "wg_reversal: " << ip.use_wg_reversal << "\n";

    std::vector<data_type> A_vec(ip.N);
    std::vector<data_type> D_vec(ip.N);

    // set up device buffers
    data_type* gpu_A;
    std::size_t bytes_A = A_vec.size() * sizeof(data_type);
    HIP_CHECK(hipMalloc(&gpu_A, bytes_A));
    //HIP_CHECK(hipMemcpy(gpu_A, A_vec.data(), bytes_A, hipMemcpyHostToDevice));

    // set up threads
    std::size_t block_size = 220;
    std::size_t grid_size = (int)ceil((float)ip.N/block_size);
    
    auto add_one_kernel = add_one<float, int>;
    hipLaunchKernelGGL(add_one_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        gpu_A,
        ip.N 
    );

    auto add_two_kernel = add_two<false, data_type, int>;
    if(ip.use_wg_reversal)
    {
        add_two_kernel = add_two<true, data_type, int>;
    }
    hipLaunchKernelGGL(add_two_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        gpu_A,
        ip.N
    );

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(D_vec.data(), gpu_A, bytes_A, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(gpu_A));
}
