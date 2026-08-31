#include "tokenizer.hpp"

#include <array>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>


namespace {

    constexpr uint32_t kTokenizerMagic = 20240328;
    constexpr uint32_t kHeaderElementCount = 256;
    constexpr uint32_t kVersionOne = 1;
    constexpr uint32_t kVersionTwo = 2;
    constexpr uint32_t kVersionOneEndOfTextToken = 50256;
    constexpr uint64_t kHeaderBytes = kHeaderElementCount * sizeof(uint32_t);

    uint8_t read_byte(std::istream& input, const std::string_view description) {
        char byte = 0;
        if (!input.get(byte)) {
            throw std::runtime_error(
                "truncated tokenizer while reading " +
                std::string{ description }
            );
        }
        return static_cast<uint8_t>(byte);
    }

    uint32_t read_uint32_le(
        std::istream& input, const std::string_view description
    ) {
        std::array<uint8_t, sizeof(uint32_t)> bytes{};
        for (uint8_t& byte : bytes) {
            byte = read_byte(input, description);
        }
        return static_cast<uint32_t>(bytes[0]) |
               static_cast<uint32_t>(bytes[1]) << 8U |
               static_cast<uint32_t>(bytes[2]) << 16U |
               static_cast<uint32_t>(bytes[3]) << 24U;
    }

    bool is_ascii_letter(const unsigned char byte) {
        return (byte >= 'A' && byte <= 'Z') || (byte >= 'a' && byte <= 'z');
    }

    bool is_ascii_digit(const unsigned char byte) {
        return byte >= '0' && byte <= '9';
    }

    bool is_ascii_whitespace(const unsigned char byte) {
        return byte == ' ' || byte == '\t' || byte == '\n' || byte == '\r' ||
               byte == '\f' || byte == '\v';
    }

    bool is_ascii_symbol(const unsigned char byte) {
        return !is_ascii_letter(byte) && !is_ascii_digit(byte) &&
               !is_ascii_whitespace(byte);
    }

    size_t consume_while(
        const std::string_view text,
        const size_t start,
        bool (*predicate)(unsigned char)
    ) {
        size_t end = start;
        while (end < text.size() &&
               predicate(static_cast<unsigned char>(text[end]))) {
            ++end;
        }
        return end;
    }

    bool starts_with_at(
        const std::string_view text,
        const size_t position,
        const std::string_view expected
    ) {
        return position <= text.size() &&
               expected.size() <= text.size() - position &&
               text.substr(position, expected.size()) == expected;
    }

}  // namespace


namespace sung {

    Gpt2Tokenizer::Gpt2Tokenizer(const Path& tokenizer_path) {
        std::ifstream input(tokenizer_path, std::ios::binary);
        if (!input) {
            throw std::runtime_error(
                "failed to open tokenizer: " + tokenizer_path.string()
            );
        }

        std::array<uint32_t, kHeaderElementCount> header{};
        for (size_t index = 0; index < header.size(); ++index) {
            header[index] = read_uint32_le(input, "header");
        }

        if (header[0] != kTokenizerMagic) {
            throw std::runtime_error("invalid tokenizer magic");
        }
        if (header[1] != kVersionOne && header[1] != kVersionTwo) {
            throw std::runtime_error(
                "unsupported tokenizer version: " + std::to_string(header[1])
            );
        }
        if (header[2] == 0) {
            throw std::runtime_error("tokenizer vocabulary is empty");
        }
        const uint64_t minimum_file_bytes = kHeaderBytes +
                                            uint64_t{ header[2] } * 2;
        if (std::filesystem::file_size(tokenizer_path) < minimum_file_bytes) {
            throw std::runtime_error(
                "truncated tokenizer vocabulary for declared token count"
            );
        }

        end_of_text_token_id_ = header[1] == kVersionOne
                                    ? kVersionOneEndOfTextToken
                                    : header[3];
        if (end_of_text_token_id_ >= header[2]) {
            throw std::runtime_error(
                "tokenizer end-of-text token is outside the vocabulary"
            );
        }

        token_bytes_.reserve(header[2]);
        for (uint32_t token_id = 0; token_id < header[2]; ++token_id) {
            const uint8_t length = read_byte(input, "token length");
            if (length == 0) {
                throw std::runtime_error(
                    "tokenizer contains an empty token at ID " +
                    std::to_string(token_id)
                );
            }

            std::string bytes(length, '\0');
            if (!input.read(
                    bytes.data(), static_cast<std::streamsize>(length)
                )) {
                throw std::runtime_error(
                    "truncated tokenizer while reading token ID " +
                    std::to_string(token_id)
                );
            }
            token_bytes_.push_back(std::move(bytes));
        }

        if (input.peek() != std::char_traits<char>::eof()) {
            throw std::runtime_error(
                "tokenizer contains trailing data after the vocabulary"
            );
        }

        token_ranks_.reserve(token_bytes_.size() - 1);
        for (uint64_t token_id = 0; token_id < token_bytes_.size();
             ++token_id) {
            if (token_id == end_of_text_token_id_) {
                continue;
            }
            const auto [iterator, inserted] = token_ranks_.emplace(
                token_bytes_[token_id], token_id
            );
            static_cast<void>(iterator);
            if (!inserted) {
                throw std::runtime_error(
                    "tokenizer contains duplicate token bytes at ID " +
                    std::to_string(token_id)
                );
            }
        }
    }

    size_t Gpt2Tokenizer::vocabulary_size() const {
        return token_bytes_.size();
    }

    uint64_t Gpt2Tokenizer::end_of_text_token_id() const {
        return end_of_text_token_id_;
    }

    std::vector<std::string> Gpt2Tokenizer::pre_tokenize(
        const std::string_view text
    ) const {
        for (const unsigned char byte : text) {
            if (byte > 0x7fU) {
                throw std::invalid_argument(
                    "non-ASCII prompts are not supported by this tokenizer yet"
                );
            }
        }

        constexpr std::array<std::string_view, 7> kContractions{
            "'s", "'t", "'re", "'ve", "'m", "'ll", "'d"
        };

        std::vector<std::string> pieces;
        size_t position = 0;
        while (position < text.size()) {
            bool found_contraction = false;
            for (const std::string_view contraction : kContractions) {
                if (starts_with_at(text, position, contraction)) {
                    pieces.emplace_back(contraction);
                    position += contraction.size();
                    found_contraction = true;
                    break;
                }
            }
            if (found_contraction) {
                continue;
            }

            const size_t category_start = text[position] == ' ' ? position + 1
                                                                : position;
            if (category_start < text.size()) {
                const unsigned char category_byte = static_cast<unsigned char>(
                    text[category_start]
                );
                bool (*predicate)(
                    unsigned char
                ) = is_ascii_letter(category_byte)   ? is_ascii_letter
                    : is_ascii_digit(category_byte)  ? is_ascii_digit
                    : is_ascii_symbol(category_byte) ? is_ascii_symbol
                                                     : nullptr;
                if (predicate != nullptr) {
                    const size_t end = consume_while(
                        text, category_start, predicate
                    );
                    pieces.emplace_back(text.substr(position, end - position));
                    position = end;
                    continue;
                }
            }

            const unsigned char byte = static_cast<unsigned char>(
                text[position]
            );
            if (!is_ascii_whitespace(byte)) {
                throw std::logic_error("ASCII pre-tokenizer made no progress");
            }

            const size_t whitespace_end = consume_while(
                text, position, is_ascii_whitespace
            );
            size_t piece_end = whitespace_end;
            if (whitespace_end < text.size() && whitespace_end - position > 1) {
                --piece_end;
            }
            pieces.emplace_back(text.substr(position, piece_end - position));
            position = piece_end;
        }
        return pieces;
    }

    void Gpt2Tokenizer::append_bpe_token_ids(
        const std::string_view piece, std::vector<uint64_t>& output
    ) const {
        const auto whole_piece = token_ranks_.find(std::string{ piece });
        if (whole_piece != token_ranks_.end()) {
            output.push_back(whole_piece->second);
            return;
        }

        std::vector<std::string> parts;
        parts.reserve(piece.size());
        for (const char byte : piece) {
            parts.emplace_back(1, byte);
        }

        while (parts.size() > 1) {
            uint64_t best_rank = std::numeric_limits<uint64_t>::max();
            std::string best_left;
            std::string best_right;
            for (size_t index = 0; index + 1 < parts.size(); ++index) {
                const auto candidate = token_ranks_.find(
                    parts[index] + parts[index + 1]
                );
                if (candidate != token_ranks_.end() &&
                    candidate->second < best_rank) {
                    best_rank = candidate->second;
                    best_left = parts[index];
                    best_right = parts[index + 1];
                }
            }
            if (best_rank == std::numeric_limits<uint64_t>::max()) {
                break;
            }

            std::vector<std::string> merged;
            merged.reserve(parts.size());
            for (size_t index = 0; index < parts.size();) {
                if (index + 1 < parts.size() && parts[index] == best_left &&
                    parts[index + 1] == best_right) {
                    merged.push_back(parts[index] + parts[index + 1]);
                    index += 2;
                } else {
                    merged.push_back(std::move(parts[index]));
                    ++index;
                }
            }
            parts = std::move(merged);
        }

        for (const std::string& part : parts) {
            const auto token = token_ranks_.find(part);
            if (token == token_ranks_.end()) {
                throw std::runtime_error(
                    "tokenizer is missing a byte token required for encoding"
                );
            }
            output.push_back(token->second);
        }
    }

    std::vector<uint64_t> Gpt2Tokenizer::encode(
        const std::string_view text
    ) const {
        std::vector<uint64_t> token_ids;
        for (const std::string& piece : pre_tokenize(text)) {
            append_bpe_token_ids(piece, token_ids);
        }
        return token_ids;
    }

    std::string Gpt2Tokenizer::decode(
        const std::span<const uint64_t> token_ids
    ) const {
        size_t output_size = 0;
        for (const uint64_t token_id : token_ids) {
            if (token_id >= token_bytes_.size()) {
                throw std::out_of_range(
                    "token ID " + std::to_string(token_id) +
                    " is outside the tokenizer vocabulary"
                );
            }
            if (token_bytes_[token_id].size() >
                std::numeric_limits<size_t>::max() - output_size) {
                throw std::overflow_error("decoded text is too large");
            }
            output_size += token_bytes_[token_id].size();
        }

        std::string output;
        output.reserve(output_size);
        for (const uint64_t token_id : token_ids) {
            output += token_bytes_[token_id];
        }
        return output;
    }

}  // namespace sung
