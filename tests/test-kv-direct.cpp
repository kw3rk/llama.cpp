#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "llama-kv-cache.h"

static void test_disabled_state() {
    printf("test_disabled_state...\n");
    llama_kv_cache::kv_direct_state state(-1);
    assert(!state.enabled);
    assert(state.budget_tokens == -1);
    printf("  PASS\n");
}

static void test_zero_budget_state() {
    printf("test_zero_budget_state...\n");
    llama_kv_cache::kv_direct_state state(0);
    assert(state.enabled);
    assert(state.budget_tokens == 0);
    printf("  PASS\n");
}

static void test_enabled_state() {
    printf("test_enabled_state...\n");
    llama_kv_cache::kv_direct_state state(100);
    assert(state.enabled);
    assert(state.budget_tokens == 100);
    printf("  PASS\n");
}

static void test_pool_store_and_lookup() {
    printf("test_pool_store_and_lookup...\n");
    llama_kv_cache::kv_direct_state state(100);
    state.pool_init(100, 4);

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    state.pool_store(5, 0, data.data());

    // Lookup existing entry
    const float * found = state.pool_lookup(5, 0);
    assert(found != nullptr);
    assert(found[0] == 1.0f);
    assert(found[1] == 2.0f);
    assert(found[2] == 3.0f);
    assert(found[3] == 4.0f);

    // Lookup wrong position
    assert(state.pool_lookup(6, 0) == nullptr);

    // Lookup wrong seq_id
    assert(state.pool_lookup(5, 1) == nullptr);

    printf("  PASS\n");
}

static void test_pool_invalidate() {
    printf("test_pool_invalidate...\n");
    llama_kv_cache::kv_direct_state state(100);
    state.pool_init(100, 4);

    std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
    state.pool_store(10, 0, data.data());

    assert(state.pool_lookup(10, 0) != nullptr);

    state.pool_invalidate(10);
    assert(state.pool_lookup(10, 0) == nullptr);

    printf("  PASS\n");
}

static void test_pool_ring_buffer_wraparound() {
    printf("test_pool_ring_buffer_wraparound...\n");
    llama_kv_cache::kv_direct_state state(8);
    state.pool_init(8, 2);

    // Fill all 8 slots
    for (int i = 0; i < 8; ++i) {
        float data[2] = {(float)i, (float)(i * 10)};
        state.pool_store(i, 0, data);
    }

    // All 8 should be findable
    for (int i = 0; i < 8; ++i) {
        assert(state.pool_lookup(i, 0) != nullptr);
    }

    // Store at pos 8 — wraps to slot 0, evicting pos 0
    float data[2] = {8.0f, 80.0f};
    state.pool_store(8, 0, data);

    // pos 0 is gone (overwritten by pos 8 in slot 0)
    assert(state.pool_lookup(0, 0) == nullptr);

    // pos 8 is present
    const float * found = state.pool_lookup(8, 0);
    assert(found != nullptr);
    assert(found[0] == 8.0f);

    printf("  PASS\n");
}

static void test_lru() {
    printf("test_lru...\n");
    llama_kv_cache::kv_direct_state state(100);
    state.lru_init(1, 10);

    // Initial step is 0
    assert(state.kv_step == 0);

    state.lru_touch(0, 3);
    assert(state.last_used[0][3] == 0);

    state.lru_step();
    assert(state.kv_step == 1);

    state.lru_touch(0, 5);
    assert(state.last_used[0][5] == 1);

    // Cell 3 still at old timestamp
    assert(state.last_used[0][3] == 0);

    printf("  PASS\n");
}

static void test_auto_mode_init() {
    printf("test_auto_mode_init...\n");

    // lazy init: auto mode starts disabled, pool allocated on first budget tightening
    llama_kv_cache::kv_direct_state state(-1);
    state.auto_mode = true;
    state.n_ctx     = 1024;

    assert(!state.enabled);
    assert(state.auto_mode);
    assert(state.budget_tokens == -1);
    assert(state.n_ctx == 1024);
    assert(state.sample_count == 0);
    assert(state.ring_used == 0);
    assert(state.pool_capacity == 0);

    printf("  PASS\n");
}

static void test_ring_buffer() {
    printf("test_ring_buffer...\n");

    llama_kv_cache::kv_direct_state state(1024);
    state.auto_mode = true;
    state.n_ctx     = 1024;

    // Fill ring buffer past capacity
    for (uint32_t i = 0; i < llama_kv_cache::kv_direct_state::RING_SIZE + 10; i++) {
        state.record_tg_sample(100 + i, 1000 + (int64_t)i * 10);
    }

    // ring_used should be capped at RING_SIZE
    assert(state.ring_used == llama_kv_cache::kv_direct_state::RING_SIZE);
    // sample_count tracks total
    assert(state.sample_count == llama_kv_cache::kv_direct_state::RING_SIZE + 10);
    // ring_head should have wrapped
    assert(state.ring_head == 10);

    // Verify the oldest sample in the ring (at position ring_head) is the 10th sample
    // (first 10 were overwritten by the last 10)
    assert(state.ring[state.ring_head].n_kv == 110);
    assert(state.ring[state.ring_head].decode_us == 1100);

    printf("  PASS\n");
}

static void test_compute_optimal_budget() {
    printf("test_compute_optimal_budget...\n");

    llama_kv_cache::kv_direct_state state(4096);
    state.auto_mode       = true;
    state.n_ctx           = 4096;
    state.min_calibration = 4;  // lower for testing

    // Not enough samples — should return current budget
    state.record_tg_sample(1000, 100);
    state.record_tg_sample(2000, 200);
    assert(state.compute_optimal_budget() == 4096);

    // Feed synthetic linear data: decode_us = 0.1 * n_kv
    // slope = 0.1 us/entry
    for (uint32_t i = 0; i < 20; i++) {
        uint32_t n_kv = 500 + i * 100;
        int64_t  us   = (int64_t)(n_kv * 0.1);
        state.record_tg_sample(n_kv, us);
    }

    // Set recompute cost: 50 us/entry
    // optimal = n_ctx - (recompute_cost / slope) = 4096 - (50 / 0.1) = 4096 - 500 = 3596
    state.recompute_cost_per_entry = 50.0f;
    assert(state.compute_optimal_budget() == 3596);

    // Test bootstrap: when recompute_cost_per_entry is 0, it gets bootstrapped from mean TG time
    llama_kv_cache::kv_direct_state state_boot(4096);
    state_boot.auto_mode       = true;
    state_boot.n_ctx           = 4096;
    state_boot.min_calibration = 4;

    // Feed data: decode_us = 10000 + n_kv * 5 (slope=5, mean_y~17250)
    // With bootstrap: recompute cost = mean_y, optimal = 4096 - (17250 / 5) = 4096 - 3450 = 646
    for (uint32_t i = 0; i < 20; i++) {
        uint32_t n_kv = 1000 + i * 100;
        int64_t  us   = 10000 + (int64_t)(n_kv * 5);
        state_boot.record_tg_sample(n_kv, us);
    }
    int32_t bootstrapped = state_boot.compute_optimal_budget();
    assert(bootstrapped > 0 && bootstrapped < 4096);  // bootstrap produced a real budget
    assert(state_boot.recompute_cost_per_entry > 0.0f);  // bootstrap set the cost

    printf("  PASS\n");
}

static void test_auto_safety_rails() {
    printf("test_auto_safety_rails...\n");

    llama_kv_cache::kv_direct_state state(4096);
    state.auto_mode       = true;
    state.n_ctx           = 4096;
    state.min_calibration = 4;

    // Feed constant timing (zero slope) — should return n_ctx
    for (uint32_t i = 0; i < 20; i++) {
        state.record_tg_sample(500 + i * 100, 1000);  // same decode_us regardless of n_kv
    }
    state.recompute_cost_per_entry = 50.0f;
    // slope should be ~0, regression will find no benefit
    // Fit should give slope = 0 => return n_ctx
    assert(state.compute_optimal_budget() == 4096);

    // Test 90% threshold: if optimal >= 0.9 * n_ctx, return n_ctx
    llama_kv_cache::kv_direct_state state2(4096);
    state2.auto_mode       = true;
    state2.n_ctx           = 4096;
    state2.min_calibration = 4;

    // slope = 1.0 us/entry, recompute_cost = 50 us/entry
    // optimal = 4096 - (50 / 1.0) = 4046 (which is > 0.9 * 4096 = 3686)
    for (uint32_t i = 0; i < 20; i++) {
        uint32_t n_kv = 500 + i * 100;
        int64_t  us   = (int64_t)n_kv;  // slope = 1.0 us/entry, clean integer data
        state2.record_tg_sample(n_kv, us);
    }
    state2.recompute_cost_per_entry = 50.0f;
    assert(state2.compute_optimal_budget() == 4096);  // 4046 > 3686, above 90% threshold

    printf("  PASS\n");
}

int main() {
    test_disabled_state();
    test_zero_budget_state();
    test_enabled_state();
    test_pool_store_and_lookup();
    test_pool_invalidate();
    test_pool_ring_buffer_wraparound();
    test_lru();
    test_auto_mode_init();
    test_ring_buffer();
    test_compute_optimal_budget();
    test_auto_safety_rails();

    printf("\nAll KV Direct tests passed!\n");
    return 0;
}
