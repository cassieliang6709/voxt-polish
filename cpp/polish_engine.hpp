#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_vocab;
struct llama_sampler;

namespace voxt {

struct AliasRule {
    std::string alias;
    std::string canonical;
    bool short_ascii = false;
};

struct EngineConfig {
    std::string model_path;
    std::string system_prompt;
    std::string dict_path;
    int32_t n_ctx = 2048;
    int32_t n_batch = 512;
    int32_t n_threads = 4;
    int32_t n_threads_batch = 8;
    int32_t n_gpu_layers = 99;
    bool flash_attn = false;
    bool enable_spec = true;
    int32_t draft_max = 6;
    bool enable_lcp_reuse = true;
    bool enable_guardrail = true;
    bool enable_postprocess = true;
};

struct PolishMetrics {
    std::string output;
    std::string raw_output;
    std::string fallback_reason;
    bool fell_back = false;
    int32_t prompt_tokens = 0;
    int32_t kv_reused_tokens = 0;
    int32_t kv_prefill_tokens = 0;
    int32_t generated_tokens = 0;
    int32_t decode_steps = 0;
    int32_t draft_proposed = 0;
    int32_t draft_accepted = 0;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double total_ms = 0.0;
    double gen_tok_s = 0.0;
};

class PolishEngine {
public:
    explicit PolishEngine(EngineConfig cfg);
    ~PolishEngine();

    PolishEngine(const PolishEngine &) = delete;
    PolishEngine &operator=(const PolishEngine &) = delete;

    // Reset cached user-suffix KV while keeping pinned system prompt KV intact.
    void reset_to_system_prefix();

    // Lightweight 1-token Metal GPU keepwarm without evicting pinned system prompt KV.
    void keepwarm_ping();

    // Polish raw ASR text (returns cleaned text + detailed latency/KV/spec metrics).
    PolishMetrics polish(const std::string &asr_text, int32_t max_tokens = -1);

    int32_t system_prompt_tokens() const { return static_cast<int32_t>(sys_tokens_.size()); }
    size_t alias_rule_count() const { return alias_rules_.size(); }
    void set_spec(bool enabled, int32_t draft_max = 6) {
        cfg_.enable_spec = enabled;
        cfg_.draft_max = draft_max;
    }
    void set_lcp_reuse(bool enabled) { cfg_.enable_lcp_reuse = enabled; }

    // Text utilities (exposed for testing & server).
    static std::string normalize_cjk_ascii_spacing(const std::string &input);
    static size_t utf8_codepoint_count(const std::string &s);
    std::string apply_alias_rules(const std::string &input, int *hits_out = nullptr) const;
    static std::string deterministic_fallback(const std::string &asr_cleaned);

private:
    std::vector<int32_t> tokenize(const std::string &text, bool parse_special) const;
    std::string token_to_piece(int32_t tok) const;
    void load_dictionary(const std::string &path);
    void init_pinned_system_kv();
    std::vector<int32_t> draft_ngram_simple(
        const std::vector<int32_t> &source_tokens,
        int32_t last_tok,
        int32_t max_draft) const;

    EngineConfig cfg_;
    llama_model *model_ = nullptr;
    llama_context *ctx_ = nullptr;
    const llama_vocab *vocab_ = nullptr;
    llama_sampler *sampler_ = nullptr;

    std::vector<int32_t> sys_tokens_;
    std::vector<int32_t> suffix_tokens_;
    std::vector<int32_t> cached_prompt_tokens_;
    int32_t kv_len_ = 0;

    std::vector<int32_t> banned_tokens_;
    std::vector<AliasRule> alias_rules_;
};

} // namespace voxt
