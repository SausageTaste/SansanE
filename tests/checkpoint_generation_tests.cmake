if(NOT EXISTS "${CHECKPOINT}" OR NOT EXISTS "${TOKENIZER}")
    message("SKIP: GPT-2 artifacts are unavailable")
    return()
endif()

# llm.c's FP32 reference selects this same sequence with greedy decoding.
# Exercise the real CLI, checkpoint layout, tokenizer, and generation loop.
execute_process(
    COMMAND "${INSPECTOR}" "${CHECKPOINT}"
        --tokenizer "${TOKENIZER}" --generate 20 --prompt "Good lord"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error
    TIMEOUT 120
)
if(NOT result STREQUAL "0")
    message(FATAL_ERROR "Generation failed (${result}): ${error}")
endif()

foreach(expected IN ITEMS
    "num_parameters: 124475904"
    "input_token_ids: 10248 15876"
    "generated_token_ids: 11 314 1101 7926 13 314 1101 7926 13 314 1101 7926 13 314 1101 7926 13 314 1101 7926"
    "complete_text: Good lord, I'm sorry. I'm sorry. I'm sorry. I'm sorry. I'm sorry"
)
    string(FIND "${output}" "${expected}\n" position)
    if(position EQUAL -1)
        message(FATAL_ERROR "Missing expected output: ${expected}")
    endif()
endforeach()

if(output MATCHES "all_finite=false")
    message(FATAL_ERROR "Generation produced non-finite activations or logits")
endif()
