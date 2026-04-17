// KV Direct benchmark: fill context with real text, measure TG throughput.
// Runs PP to fill context, then TG with greedy sampling, calling evict + recompute
// after every decode. Reports PP t/s and TG t/s.
//
// Usage:
//   llama-kv-direct-test -m model.gguf -f prompt.txt -c 65536 -t 64 \
//       --kv-budget-tokens 512 -n 128
//
// For baseline (no KV Direct), omit --kv-budget-tokens (defaults to -1).

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int64_t time_us() {
    return ggml_time_us();
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_predict = 128;

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
        prompt_tokens.resize(max_prompt);
    }

    const int n_prompt = (int) prompt_tokens.size();

    fprintf(stderr, "\n");
    fprintf(stderr, "kv-direct-test: prompt = %d tokens, generate = %d tokens, ctx = %d, kvbt = %d, kvba = %d\n",
            n_prompt, n_predict, n_ctx, params.kv_budget_tokens, params.kv_budget_auto ? 1 : 0);
    fprintf(stderr, "\n");

    // PP phase
    const int n_batch = llama_n_batch(ctx);
    fprintf(stderr, "PP: processing %d tokens in batches of %d...\n", n_prompt, n_batch);

    const int64_t t_pp_start = time_us();

    for (int i = 0; i < n_prompt; i += n_batch) {
        const int n_tokens = std::min(n_batch, n_prompt - i);
        llama_batch batch = llama_batch_get_one(prompt_tokens.data() + i, n_tokens);

        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "error: PP decode failed at offset %d\n", i);
            return 1;
        }

        llama_kv_direct_evict(ctx);
        llama_kv_direct_recompute_misses(ctx);
    }

    llama_synchronize(ctx);
    const int64_t t_pp_end = time_us();
    const double pp_ms = (double)(t_pp_end - t_pp_start) / 1000.0;
    const double pp_tps = (double)n_prompt / (pp_ms / 1000.0);

    fprintf(stderr, "PP done: %d tokens in %.1f ms (%.2f t/s)\n\n", n_prompt, pp_ms, pp_tps);

    // Greedy sampler (temp=0)
    auto sparams = llama_sampler_chain_default_params();
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // TG phase
    fprintf(stderr, "TG: generating %d tokens...\n", n_predict);

    const int64_t t_tg_start = time_us();
    int n_generated = 0;

    llama_token new_token_id;
    for (int step = 0; step < n_predict; step++) {
        new_token_id = llama_sampler_sample(smpl, ctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            fprintf(stderr, "EOS at step %d\n", step);
            break;
        }

        n_generated++;

        // Print token text to stdout
        char buf[256];
        int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            fwrite(buf, 1, n, stdout);
            fflush(stdout);
        }

        llama_batch batch = llama_batch_get_one(&new_token_id, 1);
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "error: TG decode failed at step %d\n", step);
            return 1;
        }

        llama_kv_direct_evict(ctx);
        llama_kv_direct_recompute_misses(ctx);
    }

    llama_synchronize(ctx);
    const int64_t t_tg_end = time_us();
    const double tg_ms = (double)(t_tg_end - t_tg_start) / 1000.0;
    const double tg_tps = (double)n_generated / (tg_ms / 1000.0);

    fprintf(stderr, "\n\nTG done: %d tokens in %.1f ms (%.2f t/s)\n", n_generated, tg_ms, tg_tps);

    // Summary
    fprintf(stderr, "\n");
    fprintf(stderr, "=== RESULTS ===\n");
    fprintf(stderr, "model:    %s\n", params.model.path.c_str());
    fprintf(stderr, "ctx:      %d\n", n_ctx);
    fprintf(stderr, "kvbt:     %d\n", params.kv_budget_tokens);
    fprintf(stderr, "kvba:     %d\n", params.kv_budget_auto ? 1 : 0);
    fprintf(stderr, "prompt:   %d tokens\n", n_prompt);
    fprintf(stderr, "PP:       %.2f t/s\n", pp_tps);
    fprintf(stderr, "TG:       %.2f t/s  (%d tokens)\n", tg_tps, n_generated);
    fprintf(stderr, "===============\n");

    llama_sampler_free(smpl);

    return 0;
}
