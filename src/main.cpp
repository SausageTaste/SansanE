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

        size_t head_count() const { return static_cast<size_t>(head_count_); }

        size_t layer_count() const { return static_cast<size_t>(layer_count_); }

        size_t vocabulary_size() const {
            return static_cast<size_t>(vocabulary_size_);
        }

        size_t padded_vocabulary_size() const {
            return static_cast<size_t>(padded_vocabulary_size_);
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

        std::vector<float> read_token_embedding_table(
            const sung::Path& checkpoint_path
        ) const {
            return read_tensor_elements(
                checkpoint_path, layout_[0], 0, layout_[0].element_count
            );
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

        std::vector<float> read_attention_projection_weight(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            const uint64_t elements_per_layer = checked_multiply(
                { channel_count_, channel_count_ }
            );
            return read_tensor_elements(
                checkpoint_path,
                layout_[6],
                checked_multiply({ layer_index, elements_per_layer }),
                elements_per_layer
            );
        }

        std::vector<float> read_attention_projection_bias(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            return read_tensor_row(checkpoint_path, layout_[7], layer_index);
        }

        std::vector<float> read_second_layer_norm_weight(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            return read_tensor_row(checkpoint_path, layout_[8], layer_index);
        }

        std::vector<float> read_second_layer_norm_bias(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            return read_tensor_row(checkpoint_path, layout_[9], layer_index);
        }

        std::vector<float> read_mlp_expansion_weight(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            const uint64_t elements_per_layer = checked_multiply(
                { 4, channel_count_, channel_count_ }
            );
            return read_tensor_elements(
                checkpoint_path,
                layout_[10],
                checked_multiply({ layer_index, elements_per_layer }),
                elements_per_layer
            );
        }

        std::vector<float> read_mlp_expansion_bias(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            const uint64_t elements_per_layer = checked_multiply(
                { 4, channel_count_ }
            );
            return read_tensor_elements(
                checkpoint_path,
                layout_[11],
                checked_multiply({ layer_index, elements_per_layer }),
                elements_per_layer
            );
        }

        std::vector<float> read_mlp_projection_weight(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            const uint64_t elements_per_layer = checked_multiply(
                { 4, channel_count_, channel_count_ }
            );
            return read_tensor_elements(
                checkpoint_path,
                layout_[12],
                checked_multiply({ layer_index, elements_per_layer }),
                elements_per_layer
            );
        }

        std::vector<float> read_mlp_projection_bias(
            const sung::Path& checkpoint_path, const uint64_t layer_index
        ) const {
            validate_layer_index(layer_index);
            return read_tensor_row(checkpoint_path, layout_[13], layer_index);
        }

        std::vector<float> read_final_layer_norm_weight(
            const sung::Path& checkpoint_path
        ) const {
            return read_tensor_elements(
                checkpoint_path, layout_[14], 0, layout_[14].element_count
            );
        }

        std::vector<float> read_final_layer_norm_bias(
            const sung::Path& checkpoint_path
        ) const {
            return read_tensor_elements(
                checkpoint_path, layout_[15], 0, layout_[15].element_count
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
                bytes_to_read > static_cast<uint64_t>(
                                    std::numeric_limits<std::streamsize>::max()
                                ) ||
                byte_offset > static_cast<uint64_t>(
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
                    "incomplete range from tensor " + std::string{ tensor.name }
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

    std::vector<float> linear_without_bias(
        const std::vector<float>& input,
        const std::vector<float>& weight,
        const size_t output_count
    ) {
        return linear(input, weight, std::vector<float>(output_count, 0.0F));
    }

    size_t argmax(const std::vector<float>& values) {
        if (values.empty()) {
            throw std::invalid_argument(
                "cannot find argmax of an empty vector"
            );
        }

        size_t maximum_index = 0;
        for (size_t index = 1; index < values.size(); ++index) {
            if (values[index] > values[maximum_index]) {
                maximum_index = index;
            }
        }
        return maximum_index;
    }

    std::vector<float> gelu(const std::vector<float>& input) {
        constexpr float kScalingFactor = 0.7978845608028654F;
        constexpr float kCubicCoefficient = 0.044715F;

        std::vector<float> output(input.size());
        for (size_t index = 0; index < input.size(); ++index) {
            const float value = input[index];
            const float cubic = value * value * value;
            output[index] = 0.5F * value *
                            (1.0F + std::tanh(
                                        kScalingFactor *
                                        (value + kCubicCoefficient * cubic)
                                    ));
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
            throw std::out_of_range(
                "attention head index is outside the model"
            );
        }

        const size_t channels_per_head = channels.size() / head_count;
        const size_t head_offset = head_index * channels_per_head;
        std::vector<float> output(channels_per_head);
        for (size_t channel = 0; channel < channels_per_head; ++channel) {
            output[channel] = channels[head_offset + channel];
        }
        return output;
    }

    float dot_product(
        const std::vector<float>& left, const std::vector<float>& right
    ) {
        if (left.empty() || left.size() != right.size()) {
            throw std::invalid_argument(
                "dot-product vectors must have equal, nonzero dimensions"
            );
        }

        float result = 0.0F;
        for (size_t index = 0; index < left.size(); ++index) {
            result += left[index] * right[index];
        }
        return result;
    }

    float scaled_attention_score(
        const std::vector<float>& query, const std::vector<float>& key
    ) {
        const float score = dot_product(query, key);
        const float scale = std::sqrt(static_cast<float>(query.size()));
        return score / scale;
    }

    std::vector<float> softmax(const std::vector<float>& scores) {
        if (scores.empty()) {
            throw std::invalid_argument("cannot apply softmax to no scores");
        }

        float maximum = scores.front();
        for (const float score : scores) {
            if (!std::isfinite(score)) {
                throw std::invalid_argument("softmax scores must be finite");
            }
            if (score > maximum) {
                maximum = score;
            }
        }

        std::vector<float> probabilities(scores.size());
        float exponential_sum = 0.0F;
        for (size_t index = 0; index < scores.size(); ++index) {
            probabilities[index] = std::exp(scores[index] - maximum);
            exponential_sum += probabilities[index];
        }
        for (float& probability : probabilities) {
            probability /= exponential_sum;
        }
        return probabilities;
    }

    std::vector<float> weighted_value_sum(
        const std::vector<float>& probabilities,
        const std::vector<float>& values,
        const size_t channels_per_value
    ) {
        if (probabilities.empty() || channels_per_value == 0) {
            throw std::invalid_argument(
                "attention probabilities and values must not be empty"
            );
        }
        const uint64_t expected_value_elements = checked_multiply(
            { static_cast<uint64_t>(probabilities.size()),
              static_cast<uint64_t>(channels_per_value) }
        );
        if (values.size() != expected_value_elements) {
            throw std::invalid_argument(
                "attention value dimensions do not match probabilities"
            );
        }

        std::vector<float> output(channels_per_value, 0.0F);
        for (size_t value_index = 0; value_index < probabilities.size();
             ++value_index) {
            const size_t value_offset = value_index * channels_per_value;
            for (size_t channel = 0; channel < channels_per_value; ++channel) {
                output[channel] += probabilities[value_index] *
                                   values[value_offset + channel];
            }
        }
        return output;
    }

    std::vector<float> single_token_multi_head_attention(
        const QkvVectors& qkv, const size_t head_count
    ) {
        if (qkv.query.empty() || qkv.query.size() != qkv.key.size() ||
            qkv.query.size() != qkv.value.size()) {
            throw std::invalid_argument(
                "Q, K, and V must have equal, nonzero dimensions"
            );
        }
        if (head_count == 0 || qkv.query.size() % head_count != 0) {
            throw std::invalid_argument(
                "QKV channels cannot be divided evenly into attention heads"
            );
        }

        const size_t channels_per_head = qkv.query.size() / head_count;
        std::vector<float> output(qkv.query.size());
        for (size_t head = 0; head < head_count; ++head) {
            const auto query_head = extract_attention_head(
                qkv.query, head, head_count
            );
            const auto key_head = extract_attention_head(
                qkv.key, head, head_count
            );
            const auto value_head = extract_attention_head(
                qkv.value, head, head_count
            );

            const std::vector<float> scores{
                scaled_attention_score(query_head, key_head)
            };
            const auto probabilities = softmax(scores);
            const auto head_output = weighted_value_sum(
                probabilities, value_head, channels_per_head
            );

            const size_t head_offset = head * channels_per_head;
            for (size_t channel = 0; channel < channels_per_head; ++channel) {
                output[head_offset + channel] = head_output[channel];
            }
        }
        return output;
    }

    std::vector<float> single_token_transformer_block(
        const sung::Path& checkpoint_path,
        const Gpt2Config& config,
        const std::vector<float>& input,
        const size_t layer_index
    ) {
        if (input.size() != config.channel_count()) {
            throw std::invalid_argument(
                "Transformer block input does not match channel count"
            );
        }

        // 1. GPT-2 uses pre-normalization: attention receives a normalized
        // copy while the original residual stream bypasses the operation.
        const auto first_norm_weight = config.read_first_layer_norm_weight(
            checkpoint_path, layer_index
        );
        const auto first_norm_bias = config.read_first_layer_norm_bias(
            checkpoint_path, layer_index
        );
        const auto normalized_for_attention = layer_norm(
            input, first_norm_weight, first_norm_bias
        );

        // 2. Produce Q, K, and V together. Splitting the combined projection
        // gives each attention head its query, key, and value channels.
        const auto qkv_weight = config.read_qkv_projection_weight(
            checkpoint_path, layer_index
        );
        const auto qkv_bias = config.read_qkv_projection_bias(
            checkpoint_path, layer_index
        );
        const auto combined_qkv = linear(
            normalized_for_attention, qkv_weight, qkv_bias
        );
        const auto qkv = split_qkv(combined_qkv, config.channel_count());

        // 3. Run scaled dot-product attention independently in every head and
        // concatenate the head outputs back into one channel vector.
        const auto attention = single_token_multi_head_attention(
            qkv, config.head_count()
        );

        // 4. The attention projection mixes information across heads. The
        // first residual addition preserves the block input alongside it.
        const auto attention_weight = config.read_attention_projection_weight(
            checkpoint_path, layer_index
        );
        const auto attention_bias = config.read_attention_projection_bias(
            checkpoint_path, layer_index
        );
        const auto projected_attention = linear(
            attention, attention_weight, attention_bias
        );
        const auto post_attention = add_elementwise(input, projected_attention);

        // 5. A second pre-normalization prepares the residual stream for the
        // block's feed-forward MLP without modifying the residual bypass.
        const auto second_norm_weight = config.read_second_layer_norm_weight(
            checkpoint_path, layer_index
        );
        const auto second_norm_bias = config.read_second_layer_norm_bias(
            checkpoint_path, layer_index
        );
        const auto normalized_for_mlp = layer_norm(
            post_attention, second_norm_weight, second_norm_bias
        );

        // 6. Expand 768 channels to 3072, apply GPT-2's approximate GELU
        // activation, then project the result back down to 768 channels.
        const auto expansion_weight = config.read_mlp_expansion_weight(
            checkpoint_path, layer_index
        );
        const auto expansion_bias = config.read_mlp_expansion_bias(
            checkpoint_path, layer_index
        );
        const auto expanded = linear(
            normalized_for_mlp, expansion_weight, expansion_bias
        );
        const auto activated = gelu(expanded);
        const auto projection_weight = config.read_mlp_projection_weight(
            checkpoint_path, layer_index
        );
        const auto projection_bias = config.read_mlp_projection_bias(
            checkpoint_path, layer_index
        );
        const auto projected_mlp = linear(
            activated, projection_weight, projection_bias
        );

        // 7. The second residual addition completes this Transformer block and
        // supplies the input residual stream for the following layer.
        return add_elementwise(post_attention, projected_mlp);
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

        // 4. Select one token and position for this incremental inspection. A
        // tokenizer will eventually supply token IDs for a complete sequence.
        constexpr uint64_t kInspectedTokenId = 0;
        constexpr uint64_t kInspectedPosition = 0;
        std::cout << "\ninput_token_id: " << kInspectedTokenId << '\n'
                  << "input_position: " << kInspectedPosition << '\n';

        // 5. Look up the token and position embeddings, then add them to form
        // the residual-stream input to the first Transformer block.
        const auto token_embedding = config.read_token_embedding(
            checkpoint_path, kInspectedTokenId
        );
        const auto position_embedding = config.read_position_embedding(
            checkpoint_path, kInspectedPosition
        );
        auto hidden_state = add_elementwise(
            token_embedding, position_embedding
        );
        print_vector_preview("Token embedding", token_embedding);
        print_vector_preview("Position embedding", position_embedding);
        print_vector_preview("Initial hidden state", hidden_state);

        // 6. Pass this single-token residual stream through every Transformer
        // block. Each iteration selects a different layer's parameter slices.
        for (size_t layer = 0; layer < config.layer_count(); ++layer) {
            hidden_state = single_token_transformer_block(
                checkpoint_path, config, hidden_state, layer
            );
            std::cout << "\ntransformer_layer: " << layer << '\n';
            print_vector_preview("Transformer block output", hidden_state);
        }

        // 7. Apply GPT-2's final LayerNorm after the last Transformer block.
        // This prepares the residual stream for vocabulary projection.
        const auto final_norm_weight = config.read_final_layer_norm_weight(
            checkpoint_path
        );
        const auto final_norm_bias = config.read_final_layer_norm_bias(
            checkpoint_path
        );
        const auto final_hidden_state = layer_norm(
            hidden_state, final_norm_weight, final_norm_bias
        );
        print_vector_preview(
            "Final normalized hidden state", final_hidden_state
        );

        // 8. Reuse the token-embedding table as the output matrix. Each row's
        // dot product with the hidden state becomes that token's unscaled
        // logit.
        const auto token_embedding_table = config.read_token_embedding_table(
            checkpoint_path
        );
        auto logits = linear_without_bias(
            final_hidden_state,
            token_embedding_table,
            config.padded_vocabulary_size()
        );
        logits.resize(config.vocabulary_size());
        print_vector_preview("Vocabulary logits", logits);

        // 9. Greedy selection only needs the largest logit; applying softmax
        // would change probabilities but would not change their ordering.
        const size_t next_token_id = argmax(logits);
        std::cout << "\n[Greedy next token]\n"
                  << "token_id: " << next_token_id << '\n'
                  << "logit: " << logits[next_token_id] << '\n';
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
