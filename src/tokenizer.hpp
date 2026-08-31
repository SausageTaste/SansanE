#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "auxiliary/path.hpp"


namespace sung {

    class Gpt2Tokenizer {

    public:
        explicit Gpt2Tokenizer(const Path& tokenizer_path);

        [[nodiscard]] size_t vocabulary_size() const;
        [[nodiscard]] uint64_t end_of_text_token_id() const;

        [[nodiscard]] std::vector<uint64_t> encode(std::string_view text) const;
        [[nodiscard]] std::string decode(
            std::span<const uint64_t> token_ids
        ) const;

    private:
        [[nodiscard]] std::vector<std::string> pre_tokenize(
            std::string_view text
        ) const;
        void append_bpe_token_ids(
            std::string_view piece, std::vector<uint64_t>& output
        ) const;

        std::vector<std::string> token_bytes_;
        std::unordered_map<std::string, uint64_t> token_ranks_;
        uint64_t end_of_text_token_id_ = 0;
    };

}  // namespace sung
