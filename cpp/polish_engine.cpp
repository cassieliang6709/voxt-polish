#include "polish_engine.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "llama.h"

namespace voxt {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

// Custom llama_sampler_i that hard-masks banned token IDs (e.g. <think>, </think>) to -INF
// inside the C++ sampling chain before greedy argmax selection.
struct TokenBanCtx {
    std::vector<llama_token> banned;
};

const char *ban_sampler_name(const llama_sampler *) { return "voxt-think-ban"; }

void ban_sampler_apply(llama_sampler *smpl, llama_token_data_array *cur_p) {
    auto *ctx = static_cast<TokenBanCtx *>(smpl->ctx);
    if (!ctx || ctx->banned.empty()) return;
    for (size_t i = 0; i < cur_p->size; ++i) {
        for (llama_token b : ctx->banned) {
            if (cur_p->data[i].id == b) {
                cur_p->data[i].logit = -std::numeric_limits<float>::infinity();
            }
        }
    }
}

void ban_sampler_free(llama_sampler *smpl) {
    delete static_cast<TokenBanCtx *>(smpl->ctx);
}

llama_sampler_i kTokenBanIface = {
    /* .name              = */ ban_sampler_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ ban_sampler_apply,
    /* .reset             = */ nullptr,
    /* .clone             = */ nullptr,
    /* .free              = */ ban_sampler_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

bool is_ascii_alnum(unsigned char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9');
}

bool is_pure_short_ascii(const std::string &s) {
    size_t non_space = 0;
    for (unsigned char c : s) {
        if (c == ' ') continue;
        if (!is_ascii_alnum(c)) return false;
        ++non_space;
    }
    return non_space <= 3;
}

std::string to_lower_ascii(const std::string &s) {
    std::string out = s;
    for (char &c : out) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return out;
}

// Decode one UTF-8 codepoint at byte index i, advancing i.
uint32_t next_utf8(const std::string &s, size_t &i) {
    unsigned char b0 = static_cast<unsigned char>(s[i]);
    if (b0 < 0x80) {
        ++i;
        return b0;
    } else if ((b0 & 0xE0) == 0xC0 && i + 1 < s.size()) {
        uint32_t cp = ((b0 & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        i += 2;
        return cp;
    } else if ((b0 & 0xF0) == 0xE0 && i + 2 < s.size()) {
        uint32_t cp = ((b0 & 0x0F) << 12) |
                      ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
                      (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        i += 3;
        return cp;
    } else if ((b0 & 0xF8) == 0xF0 && i + 3 < s.size()) {
        uint32_t cp = ((b0 & 0x07) << 18) |
                      ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
                      ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
                      (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        i += 4;
        return cp;
    }
    ++i;
    return b0;
}

bool is_cjk(uint32_t cp) {
    return cp >= 0x4E00 && cp <= 0x9FFF;
}

bool is_ascii_alnum_cp(uint32_t cp) {
    return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') || (cp >= '0' && cp <= '9');
}

std::string trim_ws(const std::string &s) {
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    size_t b = s.size();
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

// Minimal JSON string extractor for dictionary.json format:
// { "cat": [ ["Canonical", ["alias1", "alias2"]], ... ] }
void parse_dictionary_json(const std::string &text, std::vector<AliasRule> &rules) {
    size_t i = 0;
    auto skip_ws = [&]() {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    };
    auto parse_str = [&]() -> std::string {
        skip_ws();
        if (i >= text.size() || text[i] != '"') return "";
        ++i;
        std::string out;
        while (i < text.size() && text[i] != '"') {
            if (text[i] == '\\' && i + 1 < text.size()) {
                ++i;
                char esc = text[i++];
                if (esc == 'n') out.push_back('\n');
                else if (esc == 't') out.push_back('\t');
                else if (esc == 'r') out.push_back('\r');
                else out.push_back(esc);
            } else {
                out.push_back(text[i++]);
            }
        }
        if (i < text.size() && text[i] == '"') ++i;
        return out;
    };
    skip_ws();
    if (i >= text.size() || text[i] != '{') return;
    ++i;
    while (i < text.size()) {
        skip_ws();
        if (i < text.size() && text[i] == '}') break;
        if (i < text.size() && text[i] == ',') { ++i; continue; }
        std::string key = parse_str();
        skip_ws();
        if (i >= text.size() || text[i] != ':') break;
        ++i;
        skip_ws();
        if (i >= text.size() || text[i] != '[') break;
        ++i;
        // Array of entries: [ ["Canonical", ["a1", "a2"]], ... ]
        while (i < text.size()) {
            skip_ws();
            if (i < text.size() && text[i] == ']') { ++i; break; }
            if (i < text.size() && text[i] == ',') { ++i; continue; }
            if (text[i] != '[') break;
            ++i;
            std::string canonical = trim_ws(parse_str());
            skip_ws();
            if (i < text.size() && text[i] == ',') {
                ++i;
                skip_ws();
                if (i < text.size() && text[i] == '[') {
                    ++i;
                    while (i < text.size()) {
                        skip_ws();
                        if (i < text.size() && text[i] == ']') { ++i; break; }
                        if (i < text.size() && text[i] == ',') { ++i; continue; }
                        std::string al = trim_ws(parse_str());
                        if (al.size() >= 2 && to_lower_ascii(al) != to_lower_ascii(canonical)) {
                            rules.push_back({al, canonical, is_pure_short_ascii(al)});
                        }
                    }
                }
            }
            skip_ws();
            if (i < text.size() && text[i] == ']') ++i;
        }
    }
    std::sort(rules.begin(), rules.end(), [](const AliasRule &a, const AliasRule &b) {
        return a.alias.size() > b.alias.size();
    });
}

void quiet_log_callback(ggml_log_level level, const char *text, void *) {
    if (level >= GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

} // namespace

PolishEngine::PolishEngine(EngineConfig cfg) : cfg_(std::move(cfg)) {
    llama_log_set(quiet_log_callback, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg_.n_gpu_layers;
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP_MLOCK;

    model_ = llama_model_load_from_file(cfg_.model_path.c_str(), mparams);
    if (!model_) {
        throw std::runtime_error("Failed to load GGUF model: " + cfg_.model_path);
    }
    vocab_ = llama_model_get_vocab(model_);

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = cfg_.n_ctx;
    cparams.n_batch = cfg_.n_batch;
    cparams.n_ubatch = cfg_.n_batch;
    cparams.n_seq_max = 1;
    cparams.n_threads = cfg_.n_threads;
    cparams.n_threads_batch = cfg_.n_threads_batch;
    cparams.flash_attn_type = cfg_.flash_attn ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                              : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.no_perf = true;

    ctx_ = llama_init_from_model(model_, cparams);
    if (!ctx_) {
        llama_model_free(model_);
        model_ = nullptr;
        throw std::runtime_error("Failed to create llama_context");
    }

    // Build banned-token list (<think>, </think>) for C++ sampler-level hard masking.
    for (const char *tag : {"<think>", "</think>"}) {
        auto toks = tokenize(tag, true);
        if (toks.size() == 1) {
            banned_tokens_.push_back(toks[0]);
        }
    }

    auto sparams = llama_sampler_chain_default_params();
    sampler_ = llama_sampler_chain_init(sparams);
    auto *ban_ctx = new TokenBanCtx{banned_tokens_};
    llama_sampler_chain_add(sampler_, llama_sampler_init(&kTokenBanIface, ban_ctx));
    llama_sampler_chain_add(sampler_, llama_sampler_init_greedy());

    if (!cfg_.dict_path.empty()) {
        load_dictionary(cfg_.dict_path);
    }

    // Pre-tokenize Qwen3 chat template prefix & suffix (enable_thinking=False exact match).
    const std::string sys_block =
        "<|im_start|>system\n" + cfg_.system_prompt + "<|im_end|>\n<|im_start|>user\n";
    const std::string asst_suffix =
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
    sys_tokens_ = tokenize(sys_block, true);
    suffix_tokens_ = tokenize(asst_suffix, true);

    init_pinned_system_kv();
}

PolishEngine::~PolishEngine() {
    if (sampler_) llama_sampler_free(sampler_);
    if (ctx_) llama_free(ctx_);
    if (model_) llama_model_free(model_);
    llama_backend_free();
}

std::vector<int32_t> PolishEngine::tokenize(const std::string &text, bool parse_special) const {
    int32_t n = llama_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()),
                               nullptr, 0, false, parse_special);
    if (n < 0) n = -n;
    std::vector<int32_t> out(static_cast<size_t>(n));
    int32_t got = llama_tokenize(vocab_, text.data(), static_cast<int32_t>(text.size()),
                                 out.data(), n, false, parse_special);
    if (got < 0) {
        throw std::runtime_error("llama_tokenize failed");
    }
    out.resize(static_cast<size_t>(got));
    return out;
}

std::string PolishEngine::token_to_piece(int32_t tok) const {
    char buf[256];
    int32_t n = llama_token_to_piece(vocab_, tok, buf, sizeof(buf), 0, false);
    if (n < 0) {
        std::string big(static_cast<size_t>(-n), '\0');
        llama_token_to_piece(vocab_, tok, big.data(), -n, 0, false);
        return big;
    }
    return std::string(buf, static_cast<size_t>(n));
}

void PolishEngine::load_dictionary(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) return;
    std::ostringstream ss;
    ss << ifs.rdbuf();
    parse_dictionary_json(ss.str(), alias_rules_);
}

void PolishEngine::init_pinned_system_kv() {
    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_clear(mem, true);

    const int32_t n = static_cast<int32_t>(sys_tokens_.size());
    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int32_t i = 0; i < n; ++i) {
        batch.token[i] = sys_tokens_[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 0;
    }
    if (llama_decode(ctx_, batch) != 0) {
        llama_batch_free(batch);
        throw std::runtime_error("Failed to prefill pinned system prompt KV");
    }
    llama_batch_free(batch);
    cached_prompt_tokens_ = sys_tokens_;
    kv_len_ = n;
}

void PolishEngine::reset_to_system_prefix() {
    llama_memory_t mem = llama_get_memory(ctx_);
    const int32_t n_sys = static_cast<int32_t>(sys_tokens_.size());
    llama_memory_seq_rm(mem, 0, n_sys, -1);
    cached_prompt_tokens_ = sys_tokens_;
    kv_len_ = n_sys;
}

std::vector<int32_t> PolishEngine::draft_ngram_simple(
    const std::vector<int32_t> &source_tokens,
    int32_t last_tok,
    int32_t max_draft) const {
    std::vector<int32_t> draft;
    if (max_draft <= 0 || source_tokens.size() < 2) return draft;
    // Match the most recent occurrence of last_tok inside the user's ASR tokens (n=1, m=max_draft).
    for (int32_t idx = static_cast<int32_t>(source_tokens.size()) - 2; idx >= 0; --idx) {
        if (source_tokens[idx] == last_tok) {
            for (int32_t k = 1; k <= max_draft && idx + k < static_cast<int32_t>(source_tokens.size()); ++k) {
                int32_t cand = source_tokens[idx + k];
                if (llama_vocab_is_eog(vocab_, cand)) break;
                draft.push_back(cand);
            }
            break;
        }
    }
    return draft;
}

size_t PolishEngine::utf8_codepoint_count(const std::string &s) {
    size_t i = 0, cnt = 0;
    while (i < s.size()) {
        next_utf8(s, i);
        ++cnt;
    }
    return cnt;
}

std::string PolishEngine::normalize_cjk_ascii_spacing(const std::string &input) {
    std::istringstream iss(input);
    std::string line;
    std::vector<std::string> out_lines;
    while (std::getline(iss, line)) {
        std::string res;
        res.reserve(line.size() + 16);
        size_t i = 0;
        uint32_t prev_cp = 0;
        bool has_prev = false;
        while (i < line.size()) {
            size_t start = i;
            uint32_t cp = next_utf8(line, i);
            if (has_prev) {
                if ((is_cjk(prev_cp) && is_ascii_alnum_cp(cp)) ||
                    (is_ascii_alnum_cp(prev_cp) && is_cjk(cp))) {
                    res.push_back(' ');
                }
            }
            // Collapse repeated horizontal spaces.
            if ((cp == ' ' || cp == '\t') && !res.empty() && res.back() == ' ') {
                prev_cp = ' ';
                has_prev = true;
                continue;
            }
            if (cp == '\t') {
                res.push_back(' ');
                prev_cp = ' ';
            } else {
                res.append(line, start, i - start);
                prev_cp = cp;
            }
            has_prev = true;
        }
        out_lines.push_back(trim_ws(res));
    }
    std::string joined;
    for (size_t k = 0; k < out_lines.size(); ++k) {
        if (k > 0) joined.push_back('\n');
        joined += out_lines[k];
    }
    return trim_ws(joined);
}

std::string PolishEngine::apply_alias_rules(const std::string &input, int *hits_out) const {
    std::string cur = input;
    int hits = 0;
    for (const auto &rule : alias_rules_) {
        std::string lower_cur = to_lower_ascii(cur);
        std::string lower_al = to_lower_ascii(rule.alias);
        size_t pos = 0;
        std::string next;
        bool matched = false;
        while (true) {
            size_t found = lower_cur.find(lower_al, pos);
            if (found == std::string::npos) {
                if (matched) next.append(cur, pos, std::string::npos);
                break;
            }
            bool ok = true;
            if (rule.short_ascii) {
                if (found > 0 && is_ascii_alnum(static_cast<unsigned char>(cur[found - 1]))) ok = false;
                size_t end = found + rule.alias.size();
                if (end < cur.size() && is_ascii_alnum(static_cast<unsigned char>(cur[end]))) ok = false;
            }
            if (ok) {
                if (!matched) {
                    next.reserve(cur.size() + 32);
                    matched = true;
                }
                next.append(cur, pos, found - pos);
                next.append(rule.canonical);
                pos = found + rule.alias.size();
                ++hits;
            } else {
                if (matched) next.append(cur, pos, found - pos + 1);
                pos = found + 1;
            }
        }
        if (matched) cur = std::move(next);
    }
    if (hits_out) *hits_out = hits;
    return cur;
}

std::string PolishEngine::deterministic_fallback(const std::string &asr_cleaned) {
    return normalize_cjk_ascii_spacing(asr_cleaned);
}

PolishMetrics PolishEngine::polish(const std::string &asr_text, int32_t max_tokens) {
    const auto t_start = Clock::now();
    PolishMetrics m;

    const std::string pre_text = apply_alias_rules(asr_text);
    const std::vector<int32_t> user_tokens = tokenize(pre_text, false);

    // Build full prompt tokens: pinned sys_tokens_ + user_tokens + suffix_tokens_.
    std::vector<int32_t> prompt_tokens;
    prompt_tokens.reserve(sys_tokens_.size() + user_tokens.size() + suffix_tokens_.size());
    prompt_tokens.insert(prompt_tokens.end(), sys_tokens_.begin(), sys_tokens_.end());
    prompt_tokens.insert(prompt_tokens.end(), user_tokens.begin(), user_tokens.end());
    prompt_tokens.insert(prompt_tokens.end(), suffix_tokens_.begin(), suffix_tokens_.end());

    const int32_t n_sys = static_cast<int32_t>(sys_tokens_.size());
    const int32_t n_prompt = static_cast<int32_t>(prompt_tokens.size());
    m.prompt_tokens = n_prompt;

    if (max_tokens <= 0) {
        max_tokens = std::max<int32_t>(64, std::min<int32_t>(512, static_cast<int32_t>(user_tokens.size() * 2 + 32)));
    }
    if (n_prompt + max_tokens + cfg_.draft_max + 8 > cfg_.n_ctx) {
        max_tokens = std::max<int32_t>(32, cfg_.n_ctx - n_prompt - cfg_.draft_max - 8);
    }

    // Compute reusable KV prefix: ALWAYS >= n_sys (pinned system prompt), plus any shared
    // incremental ASR prefix from the previous live-polish call!
    int32_t n_keep = n_sys;
    if (cfg_.enable_lcp_reuse) {
        int32_t limit = std::min<int32_t>(
            static_cast<int32_t>(cached_prompt_tokens_.size()),
            std::min<int32_t>(n_prompt - 1, kv_len_));
        int32_t lcp = n_sys;
        while (lcp < limit && cached_prompt_tokens_[lcp] == prompt_tokens[lcp]) {
            ++lcp;
        }
        n_keep = lcp;
    }

    llama_memory_t mem = llama_get_memory(ctx_);
    llama_memory_seq_rm(mem, 0, n_keep, -1);
    kv_len_ = n_keep;
    m.kv_reused_tokens = n_keep;
    m.kv_prefill_tokens = n_prompt - n_keep;

    // Prefill ONLY prompt_tokens[n_keep .. n_prompt - 1]
    const auto t_prefill = Clock::now();
    {
        int32_t pos = n_keep;
        while (pos < n_prompt) {
            int32_t chunk = std::min<int32_t>(cfg_.n_batch, n_prompt - pos);
            llama_batch batch = llama_batch_init(chunk, 0, 1);
            batch.n_tokens = chunk;
            for (int32_t i = 0; i < chunk; ++i) {
                batch.token[i] = prompt_tokens[pos + i];
                batch.pos[i] = pos + i;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i] = (pos + i == n_prompt - 1) ? 1 : 0;
            }
            if (llama_decode(ctx_, batch) != 0) {
                llama_batch_free(batch);
                throw std::runtime_error("llama_decode failed during prefill");
            }
            llama_batch_free(batch);
            pos += chunk;
        }
        kv_len_ = n_prompt;
        cached_prompt_tokens_ = prompt_tokens;
    }
    m.prefill_ms = elapsed_ms(t_prefill);

    // Decode loop (with optional in-process C++ ngram-simple speculative decoding)
    const auto t_decode = Clock::now();
    llama_sampler_reset(sampler_);

    std::vector<int32_t> generated;
    generated.reserve(static_cast<size_t>(max_tokens));

    int32_t cur_tok = llama_sampler_sample(sampler_, ctx_, -1);
    int32_t cur_pos = n_prompt;
    bool hit_eos = false;

    const int32_t max_batch_tokens = 1 + (cfg_.enable_spec ? cfg_.draft_max : 0);
    llama_batch step_batch = llama_batch_init(max_batch_tokens, 0, 1);

    while (static_cast<int32_t>(generated.size()) < max_tokens) {
        if (llama_vocab_is_eog(vocab_, cur_tok)) {
            hit_eos = true;
            break;
        }
        generated.push_back(cur_tok);
        if (static_cast<int32_t>(generated.size()) >= max_tokens) {
            break;
        }

        std::vector<int32_t> draft;
        if (cfg_.enable_spec && cfg_.draft_max > 0) {
            int32_t remaining = max_tokens - static_cast<int32_t>(generated.size());
            int32_t cap = std::min<int32_t>(cfg_.draft_max, std::max<int32_t>(0, remaining - 1));
            if (cap > 0) {
                draft = draft_ngram_simple(user_tokens, cur_tok, cap);
            }
        }

        const int32_t n_draft = static_cast<int32_t>(draft.size());
        m.draft_proposed += n_draft;
        m.decode_steps += 1;

        step_batch.n_tokens = 1 + n_draft;
        step_batch.token[0] = cur_tok;
        step_batch.pos[0] = cur_pos;
        step_batch.n_seq_id[0] = 1;
        step_batch.seq_id[0][0] = 0;
        step_batch.logits[0] = 1;
        for (int32_t j = 0; j < n_draft; ++j) {
            step_batch.token[1 + j] = draft[j];
            step_batch.pos[1 + j] = cur_pos + 1 + j;
            step_batch.n_seq_id[1 + j] = 1;
            step_batch.seq_id[1 + j][0] = 0;
            step_batch.logits[1 + j] = 1;
        }

        if (llama_decode(ctx_, step_batch) != 0) {
            llama_batch_free(step_batch);
            throw std::runtime_error("llama_decode failed during generation");
        }

        // Verify cur_tok's next token (at batch idx 0), then each draft token greedily.
        int32_t next_tok = llama_sampler_sample(sampler_, ctx_, 0);
        int32_t accepted = 0;
        while (accepted < n_draft) {
            if (llama_vocab_is_eog(vocab_, next_tok)) {
                hit_eos = true;
                break;
            }
            if (next_tok != draft[accepted]) {
                break;
            }
            generated.push_back(next_tok);
            ++accepted;
            ++m.draft_accepted;
            next_tok = llama_sampler_sample(sampler_, ctx_, accepted);
        }

        // Roll back rejected draft KV entries (keep positions [0, cur_pos + 1 + accepted)).
        const int32_t new_kv_len = cur_pos + 1 + accepted;
        if (accepted < n_draft) {
            llama_memory_seq_rm(mem, 0, new_kv_len, -1);
        }
        cur_pos = new_kv_len;
        kv_len_ = new_kv_len;

        if (hit_eos) break;
        cur_tok = next_tok;
    }

    llama_batch_free(step_batch);
    m.decode_ms = elapsed_ms(t_decode);
    m.generated_tokens = static_cast<int32_t>(generated.size());
    if (m.decode_ms > 0.0 && m.generated_tokens > 0) {
        m.gen_tok_s = static_cast<double>(m.generated_tokens) / (m.decode_ms / 1000.0);
    }

    // Detokenize
    std::string raw;
    raw.reserve(generated.size() * 4);
    for (int32_t tok : generated) {
        raw += token_to_piece(tok);
    }
    raw = trim_ws(raw);
    m.raw_output = raw;

    std::string out = raw;
    if (cfg_.enable_postprocess) {
        out = apply_alias_rules(out);
        out = normalize_cjk_ascii_spacing(out);
    }

    // Circuit-breaker guardrail (matches polish_gateway.py)
    if (cfg_.enable_guardrail) {
        const size_t asr_cp = utf8_codepoint_count(pre_text);
        const size_t out_cp = utf8_codepoint_count(out);
        if (out.empty()) {
            m.fallback_reason = "empty_output";
        } else if (!hit_eos && static_cast<int32_t>(generated.size()) >= max_tokens) {
            m.fallback_reason = "runaway_max_tokens";
        } else if (out.find("<think>") != std::string::npos || out.find("</think>") != std::string::npos) {
            m.fallback_reason = "think_tag_leak";
        } else if (out_cp > std::max<size_t>(30, static_cast<size_t>(asr_cp * 1.5))) {
            m.fallback_reason = "over_expand";
        } else if (asr_cp >= 40 && out_cp < static_cast<size_t>(asr_cp * 0.55)) {
            m.fallback_reason = "under_drop";
        }
        if (!m.fallback_reason.empty()) {
            m.fell_back = true;
            out = deterministic_fallback(pre_text);
        }
    }

    m.output = out;
    m.total_ms = elapsed_ms(t_start);
    return m;
}

} // namespace voxt
