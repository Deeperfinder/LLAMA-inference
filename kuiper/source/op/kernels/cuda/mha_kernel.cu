#include <base/cuda_config.h>
#include <tensor/tensor.h>
#include <cub/cub.cuh>
#include "mha_kernel.cuh"
namespace kernel {
constexpr static int thread_num = 256;
__device__ void softmax_gpu(float* __restrict__ x, int size) {
  int tid = threadIdx.x;
  int step = blockDim.x;

  // find max value (for numerical stability)
  // this should be FLT_MAX, not 0 !!!!
  // otherwise, the softmax may be occur nan when head_dim < 128 threads
  float max_val = tid < size ? x[tid] : -FLT_MAX;
  for (int i = tid + step; i < size; i += step) {
    if (x[i] > max_val) {
      max_val = x[i];
    }
  }
  using BlockReduce = cub::BlockReduce<float, thread_num>;
  __shared__ BlockReduce::TempStorage temp;
  __shared__ float shared_val;
  max_val = BlockReduce(temp).Reduce(max_val, cub::Max());
  if (threadIdx.x == 0) {
    shared_val = max_val;
  }
  __syncthreads();
  max_val = shared_val;

  float sum = 0.0f;
  for (int i = tid; i < size; i += step) {
    x[i] = expf(x[i] - max_val);
    sum += x[i];
  }
  sum = BlockReduce(temp).Sum(sum);
  if (threadIdx.x == 0) {
    shared_val = sum;
  }
  __syncthreads();
  sum = shared_val;

  for (int i = tid; i < size; i += step) {
    x[i] /= sum;
  }
}

__global__ void multi_head_attention_kernel(int32_t pos, int32_t seq_len, float* query,
                                            float* score_ptr, float* output, float* key_cache,
                                            float* value_cache, int32_t kv_dim, int32_t kv_mul,
                                            int32_t head_num, int32_t head_size,
                                            int32_t layer_offset) {
  int head = blockIdx.x;
  if (head >= head_num) {
    return;
  }

  float scale = 1.f / sqrtf(head_size);
  float* query_head = query + head * head_size;
  // 找到score的该block head的地址
  float* score_head = score_ptr + head * seq_len;
  int head_offset = (head / kv_mul) * head_size;
  // 这里如果pos很短的话，比如0~1, 那么只有threadIdx.x = 0， 1的线程会计算
  // t×kv_dim对位到历史步和当前步长的位置
  // 1. 第一步先定位到layer位置， key_cache表示的是头地址
  // 2. 第二步定位到seq_len的位置， 即 cache = {layer_idx * seq_len * kv_dim}，
  //    seq_len的位置也就是 seq_len * kv_dim
  // 3. 第三步定位到head的位置，即 head_idx * head_size, 这里要注意 kv_dim = head_num * head_size
  for (int t = threadIdx.x; t <= pos; t += blockDim.x) {
    float* key_head = key_cache + layer_offset + t * kv_dim + head_offset;
    /**
     *  在Meta的Llama注意力机制实现中，head_dim等于head_size。
     *
     *  xq = xq.transpose(1, 2)  # 转置后形状为 (heads, sequence_length, head_dim)
     *                            # 如果sequence_length为1，则形状简化为 (heads, head_dim)
     *  keys = keys.transpose(1, 2)  # 同样转置keys，得到形状 (heads, sequence_length, head_dim)
     *                              # 若sequence_length为1，则形状也简化为 (heads, head_dim)
     *
     *  在我们的代码实现中，计算公式为 (head / kv_mul) * head_size。
     *  其中，在多头注意力（MHA）机制里，kv_mul的值为1，
     *  因此计算得到的head_offset就等于head * head_size。
     *
     *  这里的head_offset用于定位到当前处理的头部（head），而t * kv_dim (即t *
     * dim)则用于定位到历史的key向量。
     */

    // query @ key 逐个头相乘，从上面的代码可以看出
    float score = 0.0f;
#pragma unroll
    for (int i = 0; i < head_size; i += 4) {
      float4 key_head_float4 = *reinterpret_cast<float4*>(key_head + i);
      float4 query_head_float4 = *reinterpret_cast<float4*>(query_head + i);
      score += key_head_float4.x * query_head_float4.x;
      score += key_head_float4.y * query_head_float4.y;
      score += key_head_float4.z * query_head_float4.z;
      score += key_head_float4.w * query_head_float4.w;
    }

    score *= scale;
    // 第t步，即t个token的值
    score_head[t] = score;
  }
  __syncthreads();

  softmax_gpu(score_head, pos + 1);
  __syncthreads();

  float* output_head = output + head * head_size;
  // 固定value的第二维度(1+cache_len, head_size) 中的head_size, 内循环为列（seq_len）
  // 每次计算一个token，直至把所有token计算完。
  // 得到一个值， 并循环head_size次，就得到了总的 score * value
  for (int i = threadIdx.x; i < head_size; i += blockDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int t = 0; t <= pos; t++) {
      // 找到value_head第pos个token的地址
      float* value_head = value_cache + layer_offset + t * kv_dim + head_offset;
      float score = score_head[t];
      value += score * value_head[i];
    }
    output_head[i] = value;
  }
}

void mha_kernel_cu(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                   int32_t kv_dim, int32_t kv_mul, int32_t head_size, const tensor::Tensor& mha_out,
                   const tensor::Tensor& query_tensor, const tensor::Tensor& score_tensor,
                   const tensor::Tensor& key_cache_tensor, const tensor::Tensor& value_cache_tensor,
                   base::DeviceType device_type, CudaConfig* config) {
  UNUSED(device_type);
  int32_t layer_offset = layer_index * seq_len * kv_dim;
  float* query = const_cast<float*>(query_tensor.ptr<float>());
  float* score = const_cast<float*>(score_tensor.ptr<float>());
  float* output = const_cast<float*>(mha_out.ptr<float>());
  
  float* key_cache = const_cast<float*>(key_cache_tensor.ptr<float>());
  float* value_cache = const_cast<float*>(value_cache_tensor.ptr<float>());

  cudaStream_t stream = config->stream;
  multi_head_attention_kernel<<<head_num, thread_num, 0, stream>>>(
      pos, seq_len, query, score, output, key_cache, value_cache, kv_dim, kv_mul, head_num,
      head_size, layer_offset);
}

}  // namespace kernel