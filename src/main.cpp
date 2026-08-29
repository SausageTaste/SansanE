#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>


namespace {

    constexpr std::int32_t kCheckpointMagic = 20240326;
    constexpr std::int32_t kCheckpointVersion = 3;
    constexpr std::size_t kHeaderElementCount = 256;
    constexpr std::uint64_t kHeaderBytes = kHeaderElementCount *
                                           sizeof(std::int32_t);
    constexpr std::uint64_t kParameterBytes = sizeof(float);

    struct Gpt2Config {
        std::uint64_t max_sequence_length;
        std::uint64_t vocabulary_size;
        std::uint64_t padded_vocabulary_size;
        std::uint64_t layer_count;
        std::uint64_t head_count;
        std::uint64_t channel_count;
    };

    struct ParameterTensor {
        std::string_view name;
        std::vector<std::uint64_t> shape;
        std::uint64_t element_count;
        std::uint64_t byte_offset;
    };

    struct CommandLineOptions {
        std::filesystem::path checkpoint_path;
        bool list_tensors;
    };

    std::uint64_t checked_add(
        const std::uint64_t left, const std::uint64_t right
    ) {
        if (right > std::numeric_limits<std::uint64_t>::max() - left) {
            throw std::overflow_error(
                "integer overflow while calculating checkpoint size"
            );
        }
        return left + right;
    }

    std::uint64_t checked_multiply(
        const std::initializer_list<std::uint64_t> factors
    ) {
        std::uint64_t product = 1;
        for (const std::uint64_t factor : factors) {
            if (factor != 0 &&
                product > std::numeric_limits<std::uint64_t>::max() / factor) {
                throw std::overflow_error(
                    "integer overflow while calculating parameter count"
                );
            }
            product *= factor;
        }
        return product;
    }

    std::uint64_t positive_header_value(
        const std::int32_t value, const std::string_view name
    ) {
        if (value <= 0) {
            throw std::runtime_error(
                "invalid " + std::string{ name } + " in checkpoint header"
            );
        }
        return static_cast<std::uint64_t>(value);
    }

    Gpt2Config parse_config(
        const std::array<std::int32_t, kHeaderElementCount>& header
    ) {
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

        Gpt2Config config{
            .max_sequence_length = positive_header_value(
                header[2], "max_seq_len"
            ),
            .vocabulary_size = positive_header_value(header[3], "vocab_size"),
            .padded_vocabulary_size = positive_header_value(
                header[7], "padded_vocab_size"
            ),
            .layer_count = positive_header_value(header[4], "num_layers"),
            .head_count = positive_header_value(header[5], "num_heads"),
            .channel_count = positive_header_value(header[6], "channels"),
        };

        if (config.padded_vocabulary_size < config.vocabulary_size) {
            throw std::runtime_error(
                "padded_vocab_size is smaller than vocab_size"
            );
        }
        if (config.channel_count % config.head_count != 0) {
            throw std::runtime_error("channels is not divisible by num_heads");
        }

        return config;
    }

    std::vector<ParameterTensor> make_parameter_layout(
        const Gpt2Config& config
    ) {
        const std::uint64_t max_t = config.max_sequence_length;
        const std::uint64_t vocabulary = config.padded_vocabulary_size;
        const std::uint64_t layers = config.layer_count;
        const std::uint64_t channels = config.channel_count;

        struct TensorDefinition {
            std::string_view name;
            std::initializer_list<std::uint64_t> shape;
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
            TensorDefinition{ "fcprojw", { layers, channels, 4 * channels } },
            TensorDefinition{ "fcprojb", { layers, channels } },
            TensorDefinition{ "lnfw", { channels } },
            TensorDefinition{ "lnfb", { channels } },
        };

        std::vector<ParameterTensor> layout;
        layout.reserve(definitions.size());

        std::uint64_t byte_offset = kHeaderBytes;
        for (const TensorDefinition& definition : definitions) {
            const std::uint64_t elements = checked_multiply(definition.shape);
            layout.push_back(ParameterTensor{
                .name = definition.name,
                .shape = definition.shape,
                .element_count = elements,
                .byte_offset = byte_offset,
            });
            byte_offset = checked_add(
                byte_offset, checked_multiply({ elements, kParameterBytes })
            );
        }
        return layout;
    }

    std::uint64_t parameter_count(
        const std::vector<ParameterTensor>& layout
    ) {
        std::uint64_t total = 0;
        for (const ParameterTensor& tensor : layout) {
            total = checked_add(total, tensor.element_count);
        }
        return total;
    }

    std::array<std::int32_t, kHeaderElementCount> read_header(
        const std::filesystem::path& checkpoint_path
    ) {
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

        std::array<std::int32_t, kHeaderElementCount> header{};
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

    void print_parameter_layout(
        const std::vector<ParameterTensor>& layout
    ) {
        std::cout << "\n[Parameter tensors]\n"
                  << std::left << std::setw(12) << "name"
                  << std::setw(22) << "shape"
                  << std::right << std::setw(14) << "elements"
                  << std::setw(16) << "byte_offset" << '\n';

        for (const ParameterTensor& tensor : layout) {
            std::string shape;
            for (const std::uint64_t dimension : tensor.shape) {
                if (!shape.empty()) {
                    shape += " x ";
                }
                shape += std::to_string(dimension);
            }

            std::cout << std::left << std::setw(12) << tensor.name
                      << std::setw(22) << shape
                      << std::right << std::setw(14) << tensor.element_count
                      << std::setw(16) << tensor.byte_offset << '\n';
        }
    }

    void inspect_checkpoint(
        const std::filesystem::path& checkpoint_path,
        const bool list_tensors
    ) {
        const auto header = read_header(checkpoint_path);
        const Gpt2Config config = parse_config(header);
        const std::vector<ParameterTensor> layout = make_parameter_layout(
            config
        );
        const std::uint64_t parameters = parameter_count(layout);
        const std::uint64_t expected_file_bytes = checked_add(
            kHeaderBytes, checked_multiply({ parameters, kParameterBytes })
        );
        const std::uint64_t actual_file_bytes = std::filesystem::file_size(
            checkpoint_path
        );

        if (actual_file_bytes != expected_file_bytes) {
            throw std::runtime_error(
                "checkpoint size mismatch: expected " +
                std::to_string(expected_file_bytes) + " bytes, got " +
                std::to_string(actual_file_bytes)
            );
        }

        std::cout << "[GPT-2]\n"
                  << "checkpoint: " << checkpoint_path << '\n'
                  << "max_seq_len: " << config.max_sequence_length << '\n'
                  << "vocab_size: " << config.vocabulary_size << '\n'
                  << "padded_vocab_size: " << config.padded_vocabulary_size
                  << '\n'
                  << "num_layers: " << config.layer_count << '\n'
                  << "num_heads: " << config.head_count << '\n'
                  << "channels: " << config.channel_count << '\n'
                  << "num_parameters: " << parameters << '\n'
                  << "checkpoint_bytes: " << actual_file_bytes << '\n';

        if (list_tensors) {
            print_parameter_layout(layout);
        }
    }

    void print_usage(const char* executable_name) {
        std::cout
            << "Usage: " << executable_name
            << " [--list-tensors] <checkpoint_path>\n\n"
            << "Print and validate an llm.c GPT-2 v3 checkpoint.\n\n"
            << "Options:\n"
            << "  --list-tensors  Print parameter shapes and byte offsets.\n"
            << "  --help          Show this help message.\n";
    }

    std::optional<CommandLineOptions> parse_arguments(
        const int argc, char* argv[]
    ) {
        bool list_tensors = false;
        std::optional<std::filesystem::path> checkpoint_path;

        for (int index = 1; index < argc; ++index) {
            const std::string_view argument{ argv[index] };
            if (argument == "--help") {
                print_usage(argv[0]);
                return std::nullopt;
            }
            if (argument == "--list-tensors") {
                list_tensors = true;
                continue;
            }
            if (argument.starts_with('-')) {
                throw std::runtime_error(
                    "unknown option: " + std::string{ argument }
                );
            }
            if (checkpoint_path.has_value()) {
                throw std::runtime_error("multiple checkpoint paths provided");
            }
            checkpoint_path = argument;
        }

        if (!checkpoint_path.has_value()) {
            throw std::runtime_error("checkpoint path is required");
        }

        return CommandLineOptions{
            .checkpoint_path = *checkpoint_path,
            .list_tensors = list_tensors,
        };
    }

}  // namespace


int main(const int argc, char* argv[]) {
    try {
        const std::optional<CommandLineOptions> options = parse_arguments(
            argc, argv
        );
        if (!options.has_value()) {
            return 0;
        }
        inspect_checkpoint(options->checkpoint_path, options->list_tensors);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
