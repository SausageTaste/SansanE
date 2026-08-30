#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "auxiliary/path.hpp"


namespace {

    struct ParameterTensor {
        std::string_view name;
        std::vector<uint64_t> shape;
        uint64_t element_count;
        uint64_t byte_offset;
    };

    struct QkvVectors {
        std::vector<float> query;
        std::vector<float> key;
        std::vector<float> value;
    };

    uint64_t checked_add(const uint64_t left, const uint64_t right) {
        if (right > std::numeric_limits<uint64_t>::max() - left) {
            throw std::overflow_error(
                "integer overflow while calculating checkpoint size"
            );
        }
        return left + right;
    }

    uint64_t checked_multiply(const std::initializer_list<uint64_t> factors) {
        uint64_t product = 1;
        for (const uint64_t factor : factors) {
            if (factor != 0 &&
                product > std::numeric_limits<uint64_t>::max() / factor) {
                throw std::overflow_error(
                    "integer overflow while calculating parameter count"
                );
            }
            product *= factor;
        }
        return product;
    }

    uint64_t positive_header_value(
        const int32_t value, const std::string_view name
    ) {
        if (value <= 0) {
            throw std::runtime_error(
                "invalid " + std::string{ name } + " in checkpoint header"
            );
        }
        return static_cast<uint64_t>(value);
    }


    class Gpt2Config {

    public:
        constexpr static int32_t kCheckpointMagic = 20240326;
        constexpr static int32_t kCheckpointVersion = 3;
        constexpr static size_t kHeaderElementCount = 256;
        constexpr static uint64_t kHeaderBytes = kHeaderElementCount *
                                                 sizeof(int32_t);
        constexpr static uint64_t kParameterBytes = sizeof(float);

        using HeaderBuffer = std::array<int32_t, kHeaderElementCount>;

    public:
        static Gpt2Config create(const HeaderBuffer& header) {
            Gpt2Config output;
            output.parse(header);
            output.layout_ = output.make_parameter_layout();
            return output;
        }

        uint64_t parameter_count() const {
            uint64_t total = 0;
            for (auto& tensor : layout_) {
                total = checked_add(total, tensor.element_count);
            }
            return total;
        }

        uint64_t expected_file_bytes() const {
            return checked_add(
                Gpt2Config::kHeaderBytes,
                checked_multiply(
                    { this->parameter_count(), Gpt2Config::kParameterBytes }
                )
            );
        }

        size_t channel_count() const {
            return static_cast<size_t>(channel_count_);
        }

        size_t head_count() const {
            return static_cast<size_t>(head_count_);
        }

        std::vector<float> read_token_embedding(
            const sung::Path& checkpoint_path, const uint64_t token_id
        ) const {
            if (token_id >= vocabulary_size_) {
                throw std::out_of_range(
                    "token ID " + std::to_string(token_id) +
                    " is outside the vocabulary"
                );
            }
            return read_tensor_row(checkpoint_path, layout_[0], token_id);
        }

        std::vector<float> read_position_embedding(
            const sung::Path& checkpoint_path, const uint64_t position
        ) const {
            if (position >= max_sequence_length_) {
                throw std::out_of_range(
                    "position " + std::to_string(position) +
                    " is outside the maximum sequence length"
                );
            }
            return read_tensor_row(checkpoint_path, layout_[1], position);
        }

        std::vector<float> read_first_layer_norm_weight(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            return read_tensor_row(checkpoint_path, layout_[2], layer_index);
        }

        std::vector<float> read_first_layer_norm_bias(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            return read_tensor_row(checkpoint_path, layout_[3], layer_index);
        }

        std::vector<float> read_qkv_projection_weight(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            const uint64_t elements_per_layer = checked_multiply(
                { 3, channel_count_, channel_count_ }
            );
            return read_tensor_elements(
                checkpoint_path,
                layout_[4],
                checked_multiply({ layer_index, elements_per_layer }),
                elements_per_layer
            );
        }

        std::vector<float> read_qkv_projection_bias(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            const uint64_t elements_per_layer = checked_multiply(
                { 3, channel_count_ }
            );
            return read_tensor_elements(
                checkpoint_path,
                layout_[5],
                checked_multiply({ layer_index, elements_per_layer }),
                elements_per_layer
            );
        }

        friend std::ostream& operator<<(
            std::ostream& os, const Gpt2Config& config
        );

    private:
        void validate_layer_index(const uint64_t layer_index) const {
            if (layer_index >= layer_count_) {
                throw std::out_of_range(
                    "layer index " + std::to_string(layer_index) +
                    " is outside the model"
                );
            }
        }

        std::vector<float> read_tensor_row(
            const sung::Path& checkpoint_path,
            const ParameterTensor& tensor,
            const uint64_t row_index
        ) const {
            if (tensor.shape.size() != 2 || tensor.shape[1] != channel_count_) {
                throw std::runtime_error(
                    "invalid shape for tensor " + std::string{ tensor.name }
                );
            }
            if (row_index >= tensor.shape[0]) {
                throw std::out_of_range(
                    "row index is outside tensor " + std::string{ tensor.name }
                );
            }

            const uint64_t row_elements = tensor.shape[1];
            return read_tensor_elements(
                checkpoint_path,
                tensor,
                checked_multiply({ row_index, row_elements }),
                row_elements
            );
        }

        std::vector<float> read_tensor_elements(
            const sung::Path& checkpoint_path,
            const ParameterTensor& tensor,
            const uint64_t element_offset,
            const uint64_t element_count
        ) const {
            const uint64_t element_end = checked_add(
                element_offset, element_count
            );
            if (element_end > tensor.element_count) {
                throw std::out_of_range(
                    "element range is outside tensor " +
                    std::string{ tensor.name }
                );
            }

            const uint64_t bytes_to_read = checked_multiply(
                { element_count, Gpt2Config::kParameterBytes }
            );
            const uint64_t byte_offset = checked_add(
                tensor.byte_offset,
                checked_multiply(
                    { element_offset, Gpt2Config::kParameterBytes }
                )
            );

            if (element_count > std::numeric_limits<size_t>::max() ||
                bytes_to_read >
                    static_cast<uint64_t>(
                        std::numeric_limits<std::streamsize>::max()
                    ) ||
                byte_offset >
                    static_cast<uint64_t>(
                        std::numeric_limits<std::streamoff>::max()
                    )) {
                throw std::overflow_error(
                    "range from tensor " + std::string{ tensor.name } +
                    " is too large to read"
                );
            }

            std::ifstream checkpoint{ checkpoint_path, std::ios::binary };
            if (!checkpoint) {
                throw std::runtime_error(
                    "cannot open checkpoint: " + checkpoint_path.string()
                );
            }

            checkpoint.seekg(static_cast<std::streamoff>(byte_offset));
            if (!checkpoint) {
                throw std::runtime_error(
                    "cannot seek to tensor " + std::string{ tensor.name }
                );
            }

            std::vector<float> elements(static_cast<size_t>(element_count));
            const auto stream_size = static_cast<std::streamsize>(
                bytes_to_read
            );
            checkpoint.read(
                reinterpret_cast<char*>(elements.data()), stream_size
            );
            if (checkpoint.gcount() != stream_size) {
                throw std::runtime_error(
                    "incomplete range from tensor " +
                    std::string{ tensor.name }
                );
            }
            return elements;
        }

        void parse(const HeaderBuffer& header) {
            if (header[0] != kCheckpointMagic) {
                throw std::runtime_error(
                    "invalid checkpoint magic: expected " +
                    std::to_string(kCheckpointMagic) + ", got " +
                    std::to_string(header[0])
                );
            }
            if (header[1] != kCheckpointVersion) {
                throw std::runtime_error(
                    "unsupported checkpoint version: expected " +
                    std::to_string(kCheckpointVersion) + ", got " +
                    std::to_string(header[1])
                );
            }

            max_sequence_length_ = positive_header_value(
                header[2], "max_seq_len"
            );
            vocabulary_size_ = positive_header_value(header[3], "vocab_size");
            padded_vocabulary_size_ = positive_header_value(
                header[7], "padded_vocab_size"
            );
            layer_count_ = positive_header_value(header[4], "num_layers");
            head_count_ = positive_header_value(header[5], "num_heads");
            channel_count_ = positive_header_value(header[6], "channels");

            if (padded_vocabulary_size_ < vocabulary_size_) {
                throw std::runtime_error(
                    "padded_vocab_size is smaller than vocab_size"
                );
            }
            if (channel_count_ % head_count_ != 0) {
                throw std::runtime_error(
                    "channels is not divisible by num_heads"
                );
            }
        }

        std::vector<ParameterTensor> make_parameter_layout() const {
            const uint64_t max_t = max_sequence_length_;
            const uint64_t vocabulary = padded_vocabulary_size_;
            const uint64_t layers = layer_count_;
            const uint64_t channels = channel_count_;

            struct TensorDefinition {
                std::string_view name;
                std::initializer_list<uint64_t> shape;
            };

            const std::array<TensorDefinition, 16> definitions{
                TensorDefinition{ "wte", { vocabulary, channels } },
                TensorDefinition{ "wpe", { max_t, channels } },
                TensorDefinition{ "ln1w", { layers, channels } },
                TensorDefinition{ "ln1b", { layers, channels } },
                TensorDefinition{ "qkvw", { layers, 3 * channels, channels } },
                TensorDefinition{ "qkvb", { layers, 3 * channels } },
                TensorDefinition{ "attprojw", { layers, channels, channels } },
                TensorDefinition{ "attprojb", { layers, channels } },
                TensorDefinition{ "ln2w", { layers, channels } },
                TensorDefinition{ "ln2b", { layers, channels } },
                TensorDefinition{ "fcw", { layers, 4 * channels, channels } },
                TensorDefinition{ "fcb", { layers, 4 * channels } },
                TensorDefinition{ "fcprojw",
                                  { layers, channels, 4 * channels } },
                TensorDefinition{ "fcprojb", { layers, channels } },
                TensorDefinition{ "lnfw", { channels } },
                TensorDefinition{ "lnfb", { channels } },
            };

            std::vector<ParameterTensor> layout;
            layout.reserve(definitions.size());

            uint64_t byte_offset = kHeaderBytes;
            for (const TensorDefinition& definition : definitions) {
                const uint64_t elements = checked_multiply(definition.shape);
                layout.push_back(
                    ParameterTensor{
                        .name = definition.name,
                        .shape = definition.shape,
                        .element_count = elements,
                        .byte_offset = byte_offset,
                    }
                );
                byte_offset = checked_add(
                    byte_offset, checked_multiply({ elements, kParameterBytes })
                );
            }
            return layout;
        }

        uint64_t max_sequence_length_;
        uint64_t vocabulary_size_;
        uint64_t padded_vocabulary_size_;
        uint64_t layer_count_;
        uint64_t head_count_;
        uint64_t channel_count_;

        std::vector<ParameterTensor> layout_;
    };

    std::ostream& operator<<(std::ostream& os, const Gpt2Config& config) {
        os << "[GPT-2]\n"
           << "max_seq_len: " << config.max_sequence_length_ << '\n'
           << "vocab_size: " << config.vocabulary_size_ << '\n'
           << "padded_vocab_size: " << config.padded_vocabulary_size_ << '\n'
           << "num_layers: " << config.layer_count_ << '\n'
           << "num_heads: " << config.head_count_ << '\n'
           << "channels: " << config.channel_count_ << '\n'
           << "num_parameters: " << config.parameter_count() << '\n';

        os << "\n[Parameter tensors]\n"
           << std::left << std::setw(12) << "name" << std::setw(22) << "shape"
           << std::right << std::setw(14) << "elements" << std::setw(16)
           << "byte_offset" << '\n';

        for (const ParameterTensor& tensor : config.layout_) {
            std::string shape;
            for (const uint64_t dimension : tensor.shape) {
                if (!shape.empty()) {
                    shape += " x ";
                }
                shape += std::to_string(dimension);
            }

            os << std::left << std::setw(12) << tensor.name << std::setw(22)
               << shape << std::right << std::setw(14) << tensor.element_count
               << std::setw(16) << tensor.byte_offset << '\n';
        }

        return os;
    }


    Gpt2Config::HeaderBuffer read_header(const sung::Path& checkpoint_path) {
        if constexpr (std::endian::native != std::endian::little) {
            throw std::runtime_error(
                "this checkpoint reader currently requires a little-endian "
                "system"
            );
        }

        std::ifstream checkpoint{ checkpoint_path, std::ios::binary };
        if (!checkpoint) {
            throw std::runtime_error(
                "cannot open checkpoint: " + checkpoint_path.string()
            );
        }

        Gpt2Config::HeaderBuffer header{};
        checkpoint.read(
            reinterpret_cast<char*>(header.data()),
            static_cast<std::streamsize>(sizeof(header))
        );
        if (checkpoint.gcount() !=
            static_cast<std::streamsize>(sizeof(header))) {
            throw std::runtime_error(
                "checkpoint is too small to contain its header"
            );
        }
        return header;
    }

    std::vector<float> add_elementwise(
        const std::vector<float>& left, const std::vector<float>& right
    ) {
        if (left.size() != right.size()) {
            throw std::invalid_argument(
                "cannot add vectors with different dimensions"
            );
        }

        std::vector<float> result(left.size());
        for (size_t index = 0; index < result.size(); ++index) {
            result[index] = left[index] + right[index];
        }
        return result;
    }

    std::vector<float> layer_norm(
        const std::vector<float>& input,
        const std::vector<float>& weight,
        const std::vector<float>& bias,
        const float epsilon = 1e-5F
    ) {
        if (input.empty()) {
            throw std::invalid_argument("cannot normalize an empty vector");
        }
        if (input.size() != weight.size() || input.size() != bias.size()) {
            throw std::invalid_argument(
                "LayerNorm input, weight, and bias dimensions do not match"
            );
        }

        float mean = 0.0F;
        for (const float value : input) {
            mean += value;
        }
        mean /= static_cast<float>(input.size());

        float variance = 0.0F;
        for (const float value : input) {
            const float difference = value - mean;
            variance += difference * difference;
        }
        variance /= static_cast<float>(input.size());

        const float inverse_standard_deviation = 1.0F /
                                                 std::sqrt(variance + epsilon);
        std::vector<float> output(input.size());
        for (size_t index = 0; index < output.size(); ++index) {
            const float normalized = (input[index] - mean) *
                                     inverse_standard_deviation;
            output[index] = normalized * weight[index] + bias[index];
        }
        return output;
    }

    std::vector<float> linear(
        const std::vector<float>& input,
        const std::vector<float>& weight,
        const std::vector<float>& bias
    ) {
        if (input.empty() || bias.empty()) {
            throw std::invalid_argument(
                "linear input and bias must not be empty"
            );
        }
        const uint64_t expected_weight_elements = checked_multiply(
            { static_cast<uint64_t>(bias.size()),
              static_cast<uint64_t>(input.size()) }
        );
        if (weight.size() != expected_weight_elements) {
            throw std::invalid_argument(
                "linear weight dimensions do not match input and bias"
            );
        }

        std::vector<float> output = bias;
        for (size_t output_index = 0; output_index < output.size();
             ++output_index) {
            const size_t weight_row_offset = output_index * input.size();
            for (size_t input_index = 0; input_index < input.size();
                 ++input_index) {
                output[output_index] +=
                    weight[weight_row_offset + input_index] *
                    input[input_index];
            }
        }
        return output;
    }

    QkvVectors split_qkv(
        const std::vector<float>& combined, const size_t channel_count
    ) {
        const uint64_t expected_elements = checked_multiply(
            { 3, static_cast<uint64_t>(channel_count) }
        );
        if (combined.size() != expected_elements) {
            throw std::invalid_argument(
                "combined QKV dimensions do not match the channel count"
            );
        }

        QkvVectors output{
            .query = std::vector<float>(channel_count),
            .key = std::vector<float>(channel_count),
            .value = std::vector<float>(channel_count),
        };
        for (size_t channel = 0; channel < channel_count; ++channel) {
            output.query[channel] = combined[channel];
            output.key[channel] = combined[channel_count + channel];
            output.value[channel] = combined[2 * channel_count + channel];
        }
        return output;
    }

    std::vector<float> extract_attention_head(
        const std::vector<float>& channels,
        const size_t head_index,
        const size_t head_count
    ) {
        if (head_count == 0 || channels.size() % head_count != 0) {
            throw std::invalid_argument(
                "channels cannot be divided evenly into attention heads"
            );
        }
        if (head_index >= head_count) {
            throw std::out_of_range("attention head index is outside the model");
        }

        const size_t channels_per_head = channels.size() / head_count;
        const size_t head_offset = head_index * channels_per_head;
        std::vector<float> output(channels_per_head);
        for (size_t channel = 0; channel < channels_per_head; ++channel) {
            output[channel] = channels[head_offset + channel];
        }
        return output;
    }

    void print_vector_preview(
        const std::string_view title, const std::vector<float>& values
    ) {
        bool all_finite = true;
        for (const float value : values) {
            if (!std::isfinite(value)) {
                all_finite = false;
                break;
            }
        }

        constexpr size_t kPreviewElementCount = 8;
        std::cout << "\n[" << title << "]\n"
                  << "dimensions: " << values.size() << '\n'
                  << "all_finite: " << std::boolalpha << all_finite << '\n'
                  << "first_values:" << std::fixed << std::setprecision(7);
        for (size_t index = 0;
             index < values.size() && index < kPreviewElementCount;
             ++index) {
            std::cout << ' ' << values[index];
        }
        std::cout << '\n';
    }

    void inspect_checkpoint(const sung::Path& checkpoint_path) {
        // 1. Read the fixed-size header, validate the GPT-2 format, and use
        // its model dimensions to reconstruct every parameter's file offset.
        const auto header = read_header(checkpoint_path);
        const auto config = Gpt2Config::create(header);

        // 2. Verify that the reconstructed parameter layout accounts for the
        // entire file, catching truncated or incompatible checkpoints.
        const uint64_t expected_file_bytes = config.expected_file_bytes();
        const uint64_t actual_file_bytes = std::filesystem::file_size(
            checkpoint_path
        );
        if (actual_file_bytes != expected_file_bytes) {
            throw std::runtime_error(
                std::format(
                    "checkpoint size mismatch: expected {} bytes, got {}",
                    expected_file_bytes,
                    actual_file_bytes
                )
            );
        }

        // 3. Show the validated model dimensions and parameter layout before
        // reading individual weights.
        std::cout << "checkpoint: " << checkpoint_path << '\n'
                  << "checkpoint_bytes: " << actual_file_bytes << "\n\n"
                  << config;

        // 4. Select one token, position, and Transformer layer for this
        // incremental inspection. A tokenizer will eventually supply IDs.
        constexpr uint64_t kInspectedTokenId = 0;
        constexpr uint64_t kInspectedPosition = 0;
        constexpr uint64_t kInspectedLayer = 0;
        std::cout << "\ninput_token_id: " << kInspectedTokenId << '\n'
                  << "input_position: " << kInspectedPosition << '\n'
                  << "transformer_layer: " << kInspectedLayer << '\n';

        // 5. Look up the token and position embeddings, then add them to form
        // the residual-stream input to the first Transformer block.
        const auto token_embedding = config.read_token_embedding(
            checkpoint_path, kInspectedTokenId
        );
        const auto position_embedding = config.read_position_embedding(
            checkpoint_path, kInspectedPosition
        );
        const auto hidden_state = add_elementwise(
            token_embedding, position_embedding
        );
        print_vector_preview("Token embedding", token_embedding);
        print_vector_preview("Position embedding", position_embedding);
        print_vector_preview("Initial hidden state", hidden_state);

        // 6. Normalize the hidden state and apply layer 0's learned scale and
        // bias, producing the input expected by the attention projection.
        const auto layer_norm_weight = config.read_first_layer_norm_weight(
            checkpoint_path, kInspectedLayer
        );
        const auto layer_norm_bias = config.read_first_layer_norm_bias(
            checkpoint_path, kInspectedLayer
        );
        const auto normalized_hidden_state = layer_norm(
            hidden_state, layer_norm_weight, layer_norm_bias
        );
        print_vector_preview("First LayerNorm weight", layer_norm_weight);
        print_vector_preview("First LayerNorm bias", layer_norm_bias);
        print_vector_preview(
            "Normalized hidden state", normalized_hidden_state
        );

        // 7. Apply one affine projection that produces the query, key, and
        // value channels together: 768 input values become 3 * 768 values.
        const auto qkv_weight = config.read_qkv_projection_weight(
            checkpoint_path, kInspectedLayer
        );
        const auto qkv_bias = config.read_qkv_projection_bias(
            checkpoint_path, kInspectedLayer
        );
        const auto qkv = linear(
            normalized_hidden_state, qkv_weight, qkv_bias
        );
        print_vector_preview("Combined QKV projection", qkv);

        // 8. Separate Q, K, and V, then inspect one of the 12 attention heads.
        // Each head owns a contiguous 64-channel slice of all three vectors.
        const auto split = split_qkv(qkv, config.channel_count());
        constexpr size_t kInspectedHead = 0;
        const auto query_head = extract_attention_head(
            split.query, kInspectedHead, config.head_count()
        );
        const auto key_head = extract_attention_head(
            split.key, kInspectedHead, config.head_count()
        );
        const auto value_head = extract_attention_head(
            split.value, kInspectedHead, config.head_count()
        );
        std::cout << "\nattention_head: " << kInspectedHead << '\n'
                  << "channels_per_head: " << query_head.size() << '\n';
        print_vector_preview("Query head 0", query_head);
        print_vector_preview("Key head 0", key_head);
        print_vector_preview("Value head 0", value_head);
    }

}  // namespace


int main(const int argc, char* argv[]) {
    if (argc < 2)
        return -1;

    try {
        inspect_checkpoint(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
