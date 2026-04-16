#include <cassert>
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

int main() {
    test_disabled_state();
    test_zero_budget_state();
    test_enabled_state();
    test_pool_store_and_lookup();
    test_pool_invalidate();
    test_pool_ring_buffer_wraparound();
    test_lru();

    printf("\nAll KV Direct tests passed!\n");
    return 0;
}
