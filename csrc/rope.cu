#include <torch/extension.h>
#include <cuComplex.h>

constexpr size_t THREADBLOCK_SIZE = 1024;
constexpr size_t MAX_BLOCKS_X = 65535;
constexpr size_t MAX_BLOCKS_Y = 65535;



__global__ void rope_kernel(
    const float* __restrict__ sequence,
    const float* __restrict__ freqs,
    float* __restrict__ output,
    const size_t batch_size,
    const size_t seq_len,
    const size_t d_model
) {
    // input and output are both of shape (batch_size, seq_len, d_model)
    // freqs is of shape (seq_len, d_model)

    for (size_t batch_idx = blockIdx.x; batch_idx < batch_size; batch_idx += gridDim.x) {
        for (size_t seq_idx = blockIdx.y; seq_idx < seq_len; seq_idx += gridDim.y) {
            // d_idx 2 by 2
            for (size_t d_idx = threadIdx.x * 2; d_idx < d_model; d_idx += blockDim.x * 2) {
                // first, load d_idx and d_idx + 2 from sequence
                const float x1 = sequence[batch_idx * seq_len * d_model + seq_idx * d_model + d_idx];
                const float x2 = sequence[batch_idx * seq_len * d_model + seq_idx * d_model + d_idx + 1];
                // now handle x as a complex number
                const cuFloatComplex x = make_cuFloatComplex(x1, x2);
                // load the corresponding frequency
                const float f1 = freqs[(seq_idx * d_model / 2) + d_idx / 2];
                const cuFloatComplex f = make_cuFloatComplex(cosf(f1), sinf(f1)); // into polar form
                // we need to exp the frequency
                const cuFloatComplex x_rot = cuCmulf(x, f);

                output[batch_idx * seq_len * d_model + seq_idx * d_model + d_idx] = cuCrealf(x_rot);
                output[batch_idx * seq_len * d_model + seq_idx * d_model + d_idx + 1] = cuCimagf(x_rot);
            }
        }
    } 
}



torch::Tensor rope_forward(
    const torch::Tensor& sequence,
    const torch::Tensor& freqs
) { 
    torch::Tensor output = torch::empty_like(sequence);
    const size_t batch_size = sequence.size(0);
    const size_t seq_len = sequence.size(1);
    const size_t d_model = sequence.size(2);

    TORCH_CHECK(d_model % 2 == 0, "d_model must be even");


    // Check sequence and freqs have the same shape except batch_size
    TORCH_CHECK(sequence.size(1) == freqs.size(0), "sequence and freqs must have the same shape seq_len");
    TORCH_CHECK(sequence.size(2) == freqs.size(1) * 2, "sequence must have double the size of freqs in the last dimension");


    const dim3 grid_size(
        std::min(MAX_BLOCKS_X, batch_size),
        std::min(MAX_BLOCKS_Y, seq_len)
    );
    const dim3 block_size(THREADBLOCK_SIZE);

    rope_kernel<<<grid_size, block_size>>>(
        sequence.data_ptr<float>(),
        freqs.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        d_model
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("Error in rope_forward: " + std::string(cudaGetErrorString(err)));
    }

    return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_forward", &rope_forward, "Rope forward (CUDA)");
}