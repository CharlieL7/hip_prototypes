#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <hip/hip_runtime.h>

#define HIP_CHECK(command){ \
    hipError_t status = command; \
    if(status != hipSuccess) { \
        std::cerr << "Error: HIP reports " << hipGetErrorString(status) << std::endl; \
        std::abort(); \
    } \
}

/**
 * NonSeparableConvolution
 * is where each pixel of the output image
 * is the weighted sum of the neighbourhood pixels of the input image
 * The neighbourhood is defined by the dimensions of the mask and 
 * weight of each neighbour is defined by the mask itself.
 * @param input  Padded Input  matrix on which convolution is to be performed
 * @param mask   mask matrix using which convolution was to be performed
 * @param output Output matrix after performing convolution
 * @param inputDimensions dimensions of the input matrix
 * @param maskDimensions  dimensions of the mask matrix
 * @param nExWidth          Size of padded input width
 */
__global__ void simpleNonSeparableConvolution(
    unsigned int* input,
    float* mask,
    int* output,
    uint2  inputDimensions,
    uint2  maskDimensions,
    unsigned int nExWidth)
{
    unsigned int tid   = hipBlockDim_x*hipBlockIdx_x + hipThreadIdx_x;
    
    unsigned int width  = inputDimensions.x;
    unsigned int height = inputDimensions.y;
    
    unsigned int x      = tid%width;
    unsigned int y      = tid/width;
    
    unsigned int maskWidth  = maskDimensions.x;
    unsigned int maskHeight = maskDimensions.y;

    if(x >= width || y >= height)
        return;

    /*
     * initializing weighted sum value
     */
    float sumFX = 0.0f;
    int m = 0, n = 0;
  
    //performing weighted sum within the mask boundaries
    for(unsigned int j = y ; j < (y + maskHeight); ++j, m++)    
    {
        n = 0;
        for(unsigned int i = x; i < (x + maskWidth); ++i, n++)
        { 
            unsigned int maskIndex = m * maskWidth  + n;
            unsigned int index     = j * nExWidth + i;
            
            sumFX += ((float)input[index] * mask[maskIndex]);
        }
    }

    sumFX += 0.5f;
    output[tid] = (int)sumFX;
}

__global__ void vector_add(
    int* __restrict__ a,
    const int* __restrict__ b,
    const int* __restrict__ c,
    int width,
    int height) 
{
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    int i = y * width + x;
    if ( i < (width * height))
    {
        a[i] = b[i] + c[i];
    }
}


std::vector<int> ref_calc(
    const std::vector<unsigned int>& A,
    const std::vector<float>& W,
    const std::vector<int>& C,
    std::size_t height,
    std::size_t width,
    std::size_t padded_width,
    std::size_t filterSize)
{
    std::vector<int> ret(C.size());
    for(int y = 0; y < height; y++)
        for(int x = 0; x < width; x++)
        {
            float sum = 0.0f;
            for(unsigned int m = 0; m < filterSize; m++)
            {
                for(unsigned int n = 0; n < filterSize; n++)
                {
                    unsigned int maskIndex = m*filterSize+n;
                    unsigned int inputIndex = (y+m)*padded_width + (x+n);

                    // applying convolution operation
                    sum += (float)(A[inputIndex]) * (W[maskIndex]);
                }
            }
            sum += 0.5f;
            ret[(y*width + x)] = (int)sum;
        }

    std::transform(C.begin(), C.end(), ret.begin(), ret.begin(),
        [](auto c, auto d){ return c + d; });
    return ret;
}



int main(int argc, char * argv[])
{
    // doing add(conv(A, W), C)
    
    // set up buffer sizes
    std::size_t in_width = 3;
    std::size_t in_height = 3;
    std::size_t kernel_width = 3;
    std::size_t kernel_height = 3;
    std::size_t input_size = in_width * in_height;
    std::size_t kernel_size = kernel_width * kernel_height;

    
    // set up host buffers
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, 99);
    std::vector<unsigned int> A_vec(in_width * in_height);
    //std::generate(A_vec.begin(), A_vec.end(), [&](){return distrib(gen);});
    std::generate(A_vec.begin(), A_vec.end(), [&](){ return 1;});
    int kernel_radius = kernel_width / 2;
    int padded_height = in_height + kernel_height - 1;
    int padded_width = in_width + kernel_width - 1;
    std::vector<unsigned int> padded_input(padded_height * padded_width, 0);
    for(int i = 0; i < in_height; i++)
        for(int j = 0; j < in_width; j++) {
            padded_input[(i + kernel_radius) * padded_width + (j + kernel_radius)] = A_vec[i * in_width + j];
        }
    A_vec = padded_input;
    std::vector<float> W_vec = 
    {
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f,
        1.0f,  1.0f,  1.0f
    };
    std::vector<int> B_vec(input_size);
    std::vector<int> C_vec(input_size);
    std::generate(C_vec.begin(), C_vec.end(), [&](){return 1;});
    //std::generate(C_vec.begin(), C_vec.end(), [&](){return distrib(gen);});
    std::vector<int> D_vec(input_size);

    std::cout << "A_vec: [ ";
    for(auto a : A_vec)
    {
        std::cout << a << ", ";
    }
    std::cout << "]\n";
    

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

    HIP_CHECK(hipMemcpy(B_vec.data(), gpu_B, bytes_D, hipMemcpyDeviceToHost));
    std::cout << "B_vec: [ ";
    for(auto a : B_vec)
    {
        std::cout << a << ", ";
    }
    std::cout << "]\n";

    HIP_CHECK(hipMemcpy(gpu_C, C_vec.data(), bytes_C, hipMemcpyHostToDevice));

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
    HIP_CHECK(hipDeviceSynchronize());

    auto ref_vals = ref_calc(A_vec, W_vec, C_vec, in_height, in_width, padded_width, kernel_width);
    
    std::cout << "ref_vals: [ ";
    for(auto a : ref_vals)
    {
        std::cout << a << ", ";
    }
    std::cout << "]\n";

    std::cout << "D_vec: [ ";
    for(auto i : D_vec)
    {
        std::cout << i << ", ";
    }
    std::cout << "]\n";
    
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
