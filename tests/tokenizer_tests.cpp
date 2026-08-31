#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer.hpp"


namespace {

    void require(const bool condition, const std::string_view message) {
        if (!condition) {
            throw std::runtime_error(std::string{ message });
        }
    }

    void require_tokens(
        const sung::Gpt2Tokenizer& tokenizer,
        const std::string_view text,
        const std::initializer_list<uint64_t> expected
    ) {
        const auto actual = tokenizer.encode(text);
        require(
            actual == std::vector<uint64_t>{ expected },
            "unexpected tokens for: " + std::string{ text }
        );
        require(tokenizer.decode(actual) == text, "encode/decode mismatch");
    }

    template <typename Callable>
    void require_throws(
        Callable&& callable, const std::string_view expected_message
    ) {
        try {
            callable();
        } catch (const std::exception& error) {
            require(
                std::string_view{ error.what() }.find(expected_message) !=
                    std::string_view::npos,
                "exception had an unexpected message"
            );
            return;
        }
        throw std::runtime_error("expected an exception");
    }

    void write_uint32_le(std::ostream& output, const uint32_t value) {
        const std::array<char, 4> bytes{
            static_cast<char>(value & 0xffU),
            static_cast<char>((value >> 8U) & 0xffU),
            static_cast<char>((value >> 16U) & 0xffU),
            static_cast<char>((value >> 24U) & 0xffU),
        };
        output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    }

    void write_header(
        std::ostream& output,
        const uint32_t magic,
        const uint32_t version,
        const uint32_t vocabulary_size,
        const uint32_t end_of_text_token
    ) {
        write_uint32_le(output, magic);
        write_uint32_le(output, version);
        write_uint32_le(output, vocabulary_size);
        write_uint32_le(output, end_of_text_token);
        for (size_t index = 4; index < 256; ++index) {
            write_uint32_le(output, 0);
        }
    }

    void test_malformed_files() {
        const auto directory = std::filesystem::temp_directory_path();
        const auto bad_magic_path = directory /
                                    "sansane_tokenizer_bad_magic.bin";
        const auto truncated_path = directory /
                                    "sansane_tokenizer_truncated.bin";
        const auto trailing_path = directory / "sansane_tokenizer_trailing.bin";

        {
            std::ofstream output(bad_magic_path, std::ios::binary);
            write_header(output, 0, 2, 1, 0);
            output.put(1);
            output.put('a');
        }
        require_throws(
            [&] { sung::Gpt2Tokenizer tokenizer(bad_magic_path); }, "magic"
        );

        {
            std::ofstream output(truncated_path, std::ios::binary);
            write_header(output, 20240328, 2, 1, 0);
            output.put(2);
            output.put('a');
        }
        require_throws(
            [&] { sung::Gpt2Tokenizer tokenizer(truncated_path); }, "truncated"
        );

        {
            std::ofstream output(trailing_path, std::ios::binary);
            write_header(output, 20240328, 2, 1, 0);
            output.put(1);
            output.put('a');
            output.put('x');
        }
        require_throws(
            [&] { sung::Gpt2Tokenizer tokenizer(trailing_path); }, "trailing"
        );

        std::filesystem::remove(bad_magic_path);
        std::filesystem::remove(truncated_path);
        std::filesystem::remove(trailing_path);
    }

}  // namespace


int main(const int argc, char* argv[]) {
    try {
        test_malformed_files();

        if (argc != 2 || !std::filesystem::exists(argv[1])) {
            std::cout << "SKIP: GPT-2 tokenizer artifact is unavailable\n";
            return 77;
        }

        const sung::Gpt2Tokenizer tokenizer(argv[1]);
        require(tokenizer.vocabulary_size() == 50257, "wrong vocabulary size");
        require(
            tokenizer.end_of_text_token_id() == 50256, "wrong end-of-text token"
        );

        require_tokens(tokenizer, "Hello, world!", { 15496, 11, 995, 0 });
        require_tokens(tokenizer, "hello world", { 31373, 995 });
        require_tokens(tokenizer, " Hello", { 18435 });
        require_tokens(
            tokenizer, "I'm testing 123.", { 40, 1101, 4856, 17031, 13 }
        );
        require_tokens(tokenizer, "a  b", { 64, 220, 275 });
        require_tokens(tokenizer, "a \n b", { 64, 220, 198, 275 });
        require_tokens(tokenizer, "\tHello\n", { 197, 15496, 198 });
        require_tokens(tokenizer, "  hello", { 220, 23748 });
        require_tokens(tokenizer, "hello   ", { 31373, 220, 220, 220 });
        require_tokens(
            tokenizer,
            "can't won't we've I'd he'll",
            { 5171, 470, 1839, 470, 356, 1053, 314, 1549, 339, 1183 }
        );
        require_tokens(tokenizer, "ABC123!!!", { 24694, 10163, 10185 });
        require_tokens(tokenizer, "IT'S", { 2043, 6, 50 });
        require_tokens(tokenizer, "foo--bar", { 21943, 438, 5657 });
        require_tokens(
            tokenizer, "<|endoftext|>", { 27, 91, 437, 1659, 5239, 91, 29 }
        );
        require(tokenizer.encode("").empty(), "empty text should encode empty");

        require_throws(
            [&] { static_cast<void>(tokenizer.encode("caf\xc3\xa9")); },
            "non-ASCII"
        );
        const std::array<uint64_t, 1> invalid_token{ 50257 };
        require_throws(
            [&] { static_cast<void>(tokenizer.decode(invalid_token)); },
            "outside"
        );
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }

    std::cout << "PASS: tokenizer tests\n";
    return 0;
}
