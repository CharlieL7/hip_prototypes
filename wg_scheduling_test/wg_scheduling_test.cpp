#include <random>
#include <cstdlib>
#include <iostream>
#include <hip/hip_runtime.h>
#include "argparse.hpp"
#include "kernels.hpp"

using data_type = float;

struct input_params
{
	int N = 0;
    bool use_wg_reversal = false;
	int block_size = 0;
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
    program.add_argument("--block-size")
        .help("number of threads in a block, defaults to 256")
        .default_value(256)
        .scan<'i', int>();
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
        program.get<bool>("--wg-rev"),
        program.get<int>("--block-size")
    };
    return ip;
}

int main(int argc, char * argv[])
{
    auto ip = parse_inputs(argc, argv);

    // set up threads
    std::size_t block_size = ip.block_size;
    std::size_t grid_size = (int)ceil((float)ip.N/block_size);

    std::cout << "N: " << ip.N << "\n";
    std::cout << "wg_reversal: " << ip.use_wg_reversal << "\n";
    std::cout << "block_size: " << block_size << "\n";
    std::cout << "grid_size: " << grid_size << "\n";

    std::vector<data_type> A_vec(ip.N);
    std::vector<data_type> D_vec(ip.N);
    std::vector<unsigned> wg_data(grid_size);
    std::vector<long long int> timer_data(grid_size);

    // set up device buffers
    data_type* gpu_A;
    unsigned* gpu_wg_data;
    long long int* gpu_timer_data;
    std::size_t bytes_A = A_vec.size() * sizeof(data_type);
    std::size_t bytes_wg_data = wg_data.size() * sizeof(unsigned);
    std::size_t bytes_timer_data = timer_data.size() * sizeof(long long int);
    HIP_CHECK(hipMalloc(&gpu_A, bytes_A));
    HIP_CHECK(hipMalloc(&gpu_wg_data, bytes_wg_data));
    HIP_CHECK(hipMalloc(&gpu_timer_data, bytes_timer_data));
    
    auto set_to_zero_kernel = set_to_zero_debug<float, int>;
    hipLaunchKernelGGL(set_to_zero_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        gpu_A,
        gpu_wg_data,
        gpu_timer_data,
        ip.N 
    );

    auto add_two_kernel = add_two_debug<false, data_type, int>;
    if(ip.use_wg_reversal)
    {
        add_two_kernel = add_two_debug<true, data_type, int>;
    }
    hipLaunchKernelGGL(add_two_kernel,
        dim3(grid_size),
        dim3(block_size),
        0,
        0,
        gpu_A,
        gpu_wg_data,
        gpu_timer_data,
        ip.N
    );

    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(D_vec.data(), gpu_A, bytes_A, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wg_data.data(), gpu_wg_data, bytes_wg_data, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(timer_data.data(), gpu_timer_data, bytes_timer_data, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(gpu_A));
    HIP_CHECK(hipFree(gpu_wg_data));
    HIP_CHECK(hipFree(gpu_timer_data));
    
    // print data as a csv
    std::cout << "wall_time, blockIdx.x, xcc_id, se_id, cu_id\n";
    for(int i = 0; i < grid_size; ++i)
    {
        unsigned smid = wg_data[i];
        unsigned xcc_id = smid >> 6;
        unsigned se_id = smid & 0x30 >> 4;
        unsigned cu_id = smid & 0xf;
        std::cout << timer_data[i] << ", " << i << ", " << xcc_id << ", " << se_id << ", " << cu_id << "\n";
    }

    return 0;
}
