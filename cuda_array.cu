#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cstring>

#include <array>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

template <typename T>
struct CudaArray
{
  T *device_;
  T byte_size_;

  CudaArray(){};
  
  template <typename container>
  CudaArray(const container &host_input);

  void allocate(const T &size);

  template <typename container>
  void copy_to_device(const container &host_input);

  template <typename container>
  void copy_to_host(container &host_input);

  T* operator()()
  {
    return device_;
  }

  ~CudaArray()
  {
    cudaFree(device_);
  }
};

template <typename T>
void CudaArray<T>::allocate(const T &size)
{
  byte_size_ = size * sizeof(T);
  gpuErrchk(cudaMalloc((void**)&device_, byte_size_));
}

template <typename T>
template <typename container>
void CudaArray<T>::copy_to_device(const container &host_input)
{
  gpuErrchk(cudaMemcpy(this->device_, host_input.data(),
    this->byte_size_, cudaMemcpyHostToDevice));
}

template <typename T>
template <typename container>
void CudaArray<T>::copy_to_host(container &host_input)
{
  gpuErrchk(cudaMemcpy(host_input.data(), this->device_,
    this->byte_size_, cudaMemcpyDeviceToHost));
}

template <typename T>
template <typename container>
CudaArray<T>::CudaArray(const container &host_input)
{
  this->allocate(host_input.size());
  this->copy_to_device(host_input);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

int main(void)
{
  const int size = 1 << 27, block_size = 128;
  const dim3 block(block_size), grid((size / block.x) + 1);
  
  std::array<int, size> h_input;
  std::generate(h_input.begin(), h_input.end(), []{ return rand() % size; });

  CudaArray<int> gpu_arr(h_input);

  auto cpu_sum = std::accumulate(h_input.begin(), h_input.end(), 0);


  cudaDeviceSynchronize();

  gpu_arr.copy_to_host(h_input);

  cudaDeviceReset();
  return 0;
}
