#ifndef  CONV_ADD_HPP
#define CONV_ADD_HPP

#include <hip/hip_runtime.h>
#include <algorithm>
#include <vector>

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
    std::vector<int> C,
    std::size_t out_height,
    std::size_t out_width,
    std::size_t padded_width,
    std::size_t filterSize)
{
    std::vector<int> ret(C.size());
    for(int y = 0; y < out_height; y++)
        for(int x = 0; x < out_width; x++)
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
            ret[(y * out_width + x)] = (int)sum;
        }

    std::transform(C.begin(), C.end(), ret.begin(), ret.begin(),
        [](auto c, auto d){ return c + d; });
    return ret;
}
#endif
