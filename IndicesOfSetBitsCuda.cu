#include <cassert>
#include <cstdio>

#include "IndicesOfSetBits.h"

void checkCuda(cudaError_t result) {
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    abort();
  }
}

constexpr int kByteCount = 1 << 20;
constexpr int kRepsCount = 100;
constexpr int kBlockSize = 256;
constexpr int kBlockCount = kByteCount / (sizeof(uint32_t) * kBlockSize * 2);

__device__ uint32_t exclusiveScan(uint32_t* a) {
  int d = 1;
  while (d <= blockDim.x) {
    int i = (threadIdx.x + 1) * d * 2 - 1;
    if (i < 2 * blockDim.x) {
      a[i] += a[i - d];
    }
    __syncthreads();
    d *= 2;
  }
  uint32_t ans = 0;
  if (threadIdx.x == 0) {
    ans = a[2 * blockDim.x - 1];
    a[2 * blockDim.x - 1] = 0;
  }
  __syncthreads();
  d = blockDim.x;
  while (d > 0) {
    int i = (threadIdx.x + 1) * d * 2 - 1;
    if (i < 2 * blockDim.x) {
      auto tmp = a[i];
      a[i] += a[i - d];
      a[i - d] = tmp;
    }
    __syncthreads();
    d /= 2;
  }
  return ans;
}

__global__ void computePositions1(const uint32_t* bits, uint32_t* blockSums, uint32_t* positions) {
  __shared__ uint32_t pos[2 * kBlockSize];
  auto i = 2 * blockIdx.x * blockDim.x + threadIdx.x;
  pos[threadIdx.x] = __popc(bits[i]);
  pos[threadIdx.x + kBlockSize] = __popc(bits[i + kBlockSize]);
  __syncthreads();
  auto sum = exclusiveScan(pos);
  if (threadIdx.x == 0) {
    blockSums[blockIdx.x] = sum;
  }
  positions[i] = pos[threadIdx.x];
  positions[i + kBlockSize] = pos[threadIdx.x + kBlockSize];
}

__global__ void computePositions2(uint32_t* blockSums, int* outCount) {
  __shared__ uint32_t a[kBlockCount];
  auto i = threadIdx.x;
  a[i] = blockSums[i];
  a[i + blockDim.x] = blockSums[i + blockDim.x];
  __syncthreads();
  auto sum = exclusiveScan(a);
  if (threadIdx.x == 0) {
    *outCount = sum;
  }
  blockSums[i] = a[i];
  blockSums[i + blockDim.x] = a[i + blockDim.x];
}

template <bool kWriteGlobal>
__global__ void writeResult(const uint32_t* bits, const uint32_t* blockSums, const uint32_t* positions, int32_t* indices) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockSums[blockIdx.x / 2] + positions[i];
  auto w = bits[i];
  int32_t ans{};
  while (w) {
    if constexpr (kWriteGlobal) {
      indices[j++] = i * 32 + __ffs(w) - 1;
    } else {
      ans ^= i * 32 + __ffs(w) - 1;
    }
    w &= w - 1;
  }
  if constexpr (!kWriteGlobal) {
    if (i == 0) {
      indices[0] = ans;
    }
  }
}

struct IndicesOfSetBitsCuda {
  IndicesOfSetBitsCuda(const void* bits) {
    checkCuda(cudaMalloc(&bits_, kByteCount));
    checkCuda(cudaMalloc(&indices_, sizeof(int32_t) * 8 * kByteCount));
    checkCuda(cudaMalloc(&blockSums_, sizeof(uint32_t) * kBlockCount));
    checkCuda(cudaMalloc(&positions_, kByteCount));
    checkCuda(cudaMalloc(&count_, sizeof(int)));
    checkCuda(cudaMemcpy(bits_, bits, kByteCount, cudaMemcpyHostToDevice));
    checkCuda(cudaMallocHost(&actual_, sizeof(int32_t) * 8 * kByteCount));
  }

  void run() {
    computePositions1<<<kBlockCount, kBlockSize>>>(bits_, blockSums_, positions_);
    // Invoke a separate kernel to synchronize different blocks.
    static_assert(kBlockCount % 2 == 0);
    computePositions2<<<1, kBlockCount / 2>>>(blockSums_, count_);
    writeResult<true><<<2*kBlockCount, kBlockSize>>>(bits_, blockSums_, positions_, indices_);
  }

  void validate(int expectedCount, const int32_t* expected) {
    int actualCount;
    checkCuda(cudaMemcpy(&actualCount, count_, sizeof(int), cudaMemcpyDeviceToHost));
    assert(actualCount == expectedCount);
    checkCuda(cudaMemcpy(actual_, indices_, sizeof(int32_t) * actualCount, cudaMemcpyDeviceToHost));
    for (int i = 0; i < actualCount; ++i) {
      assert(actual_[i] == expected[i]);
    }
  }

private:
  // Device memory
  uint32_t* bits_;
  int* count_;
  int32_t* indices_;
  uint32_t* blockSums_;
  uint32_t* positions_;

  // Host memory
  int32_t* actual_; // too large to put on stack
};

char bits[kByteCount];
int32_t expected[8 * kByteCount];

int main() {
  for (int i = 0; i < kByteCount; ++i) {
    char byte{};
    for (int j = 0; j < 8; ++j) {
      byte |= ((rand() % 100) < 50) << j;
    }
    bits[i] = byte;
  }

  cudaEvent_t startEvent, stopEvent;
  float ms;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  int expectedCount;
  indicesOfSetBits(reinterpret_cast<const uint64_t*>(bits), 8 * kByteCount, expectedCount, expected);
  checkCuda(cudaEventRecord(startEvent));
  for (int i = 0; i < kRepsCount; ++i) {
    int n;
    indicesOfSetBits(reinterpret_cast<const uint64_t*>(bits), 8 * kByteCount, n, expected);
    assert(n == expectedCount);
  }
  checkCuda(cudaEventRecord(stopEvent));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("CPU Implementation: %.2f ms\n", ms);

  // checkCuda(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  IndicesOfSetBitsCuda runner(bits);
  runner.run();
  runner.validate(expectedCount, expected);
  checkCuda(cudaEventRecord(startEvent));
  for (int i = 0; i < kRepsCount; ++i) {
    runner.run();
  }
  checkCuda(cudaEventRecord(stopEvent));
  checkCuda(cudaEventSynchronize(stopEvent));
  checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
  printf("GPU Implementation: %.2f ms\n", ms);

  return 0;
}
