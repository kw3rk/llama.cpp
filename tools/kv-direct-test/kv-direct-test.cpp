// KV Direct accuracy test: PP a long prompt, then TG with greedy sampling.
// After every decode (PP and TG), call evict + recompute.
// Prints TSV of (step, token_id, top_logit, token_text) for diffing across budgets.
//
// Usage:
//   llama-kv-direct-test -m model.gguf -f prompt.txt -c 4096 -ngl 80 \
//       --kv-budget-tokens 512 -n 256

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 256;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_PERPLEXITY)) {
        return 1;
    }

    common_init();

    auto llama_init = common_init_from_params(params);
    llama_model   * model = llama_init->model();
    llama_context * ctx   = llama_init->context();

    if (!model || !ctx) {
        fprintf(stderr, "error: failed to init model/context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_ctx   = llama_n_ctx(ctx);

    // Read and tokenize the prompt file
    if (params.prompt_file.empty() && params.prompt.empty()) {
        fprintf(stderr, "error: provide a prompt with -f <file> or -p <text>\n");
        return 1;
    }

    std::string prompt_text = params.prompt;
    if (!params.prompt_file.empty()) {
        // common_params_parse with -f already loads the file into params.prompt
        prompt_text = params.prompt;
    }

    if (prompt_text.empty()) {
        fprintf(stderr, "error: prompt is empty\n");
        return 1;
    }

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, prompt_text, true, true);

    // Truncate prompt to leave room for generation
    const int n_predict = params.n_predict;
    const int max_prompt = n_ctx - n_predict;
    if (max_prompt <= 0) {
        fprintf(stderr, "error: n_ctx (%d) too small for n_predict (%d)\n", n_ctx, n_predict);
        return 1;
    }
    if ((int) prompt_tokens.size() > max_prompt) {
        fprintf(stderr, "truncating prompt from %zu to %d tokens (ctx=%d, predict=%d)\n",
                prompt_tokens.size(), max_prompt, n_ctx, n_predict);
        prompt_tokens.resize(max_prompt);
    }

    fprintf(stderr, "prompt: %zu tokens, generating %d tokens, ctx=%d, kvbt=%d\n",
            prompt_tokens.size(), n_predict, n_ctx, params.kv_budget_tokens);

    // PP phase: decode prompt in batches
    const int n_batch = llama_n_batch(ctx);
    fprintf(stderr, "PP: processing %zu tokens in batches of %d...\n", prompt_tokens.size(), n_batch);

    for (int i = 0; i < (int) prompt_tokens.size(); i += n_batch) {
        const int n_tokens = std::min(n_batch, (int) prompt_tokens.size() - i);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, n_tokens);

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "error: PP decode failed at offset %d\n", i);
            return 1;
        }

        llama_kv_direct_evict(ctx);
        llama_kv_direct_recompute_misses(ctx);
    }

    fprintf(stderr, "PP done. Starting TG...\n");

    // Greedy sampler
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // TG phase: generate tokens one by one
    printf("step\ttoken_id\tlogit\ttoken\n");

    llama_token new_token_id;
    for (int step = 0; step < n_predict; step++) {
        // Sample from last decode's logits
        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            fprintf(stderr, "EOS at step %d\n", step);
            break;
        }

        // Get the logit value for the chosen token
        const float * logits = llama_get_logits(ctx);
        float top_logit = logits[new_token_id];

        // Token text
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        std::string token_str(buf, n > 0 ? n : 0);

        // Escape tabs and newlines for TSV
        for (auto & c : token_str) {
            if (c == '\t') c = ' ';
            if (c == '\n') c = ' ';
            if (c == '\r') c = ' ';
        }

        printf("%d\t%d\t%.6f\t%s\n", step, new_token_id, top_logit, token_str.c_str());
        fflush(stdout);

        // Decode the new token
        llama_batch batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "error: TG decode failed at step %d\n", step);
            return 1;
        }

        llama_kv_direct_evict(ctx);
        llama_kv_direct_recompute_misses(ctx);
    }

    fprintf(stderr, "\nDone.\n");

    llama_sampler_free(smpl);

    return 0;
}
