#include "polish_engine.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string read_file(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) throw std::runtime_error("Cannot open file: " + path);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

std::string json_escape(const std::string &s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

// Extract string value for a JSON key (supports UTF-8 & standard escapes + \uXXXX).
bool extract_json_string(const std::string &json, const std::string &key, std::string &out, size_t start_pos = 0) {
    const std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle, start_pos);
    if (pos == std::string::npos) return false;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) ++pos;
    if (pos >= json.size() || json[pos] != '"') return false;
    ++pos;
    out.clear();
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            ++pos;
            char e = json[pos++];
            if (e == 'n') out.push_back('\n');
            else if (e == 'r') out.push_back('\r');
            else if (e == 't') out.push_back('\t');
            else if (e == 'u' && pos + 4 <= json.size()) {
                unsigned int cp = 0;
                std::sscanf(json.c_str() + pos, "%4x", &cp);
                pos += 4;
                if (cp < 0x80) {
                    out.push_back(static_cast<char>(cp));
                } else if (cp < 0x800) {
                    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                } else {
                    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
                    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
                    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
                }
            } else {
                out.push_back(e);
            }
        } else {
            out.push_back(json[pos++]);
        }
    }
    return true;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2 == 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t idx = std::min<size_t>(v.size() - 1, static_cast<size_t>(p * static_cast<double>(v.size() - 1)));
    return v[idx];
}

// Extract last user message content from OpenAI /v1/chat/completions JSON body.
std::string extract_last_user_content(const std::string &body) {
    size_t pos = 0;
    std::string last_user;
    while (true) {
        size_t rpos = body.find("\"role\"", pos);
        if (rpos == std::string::npos) break;
        std::string role;
        if (!extract_json_string(body, "role", role, rpos)) break;
        if (role == "user") {
            std::string content;
            if (extract_json_string(body, "content", content, rpos)) {
                last_user = content;
            }
        }
        pos = rpos + 6;
    }
    return last_user;
}

// Strip Utter's <input>...</input> wrapper if present; return whether Utter XML mode is active.
bool unwrap_utter_input(std::string &text) {
    size_t a = text.find("<input>");
    size_t b = text.rfind("</input>");
    if (a != std::string::npos && b != std::string::npos && b > a + 7) {
        text = text.substr(a + 7, b - (a + 7));
        while (!text.empty() && (text.front() == '\n' || text.front() == '\r' || text.front() == ' '))
            text.erase(text.begin());
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
            text.pop_back();
        return true;
    }
    return false;
}

void send_all(int fd, const std::string &data) {
    size_t sent = 0;
    while (sent < data.size()) {
        ssize_t n = ::send(fd, data.data() + sent, data.size() - sent, 0);
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
}

void run_http_server(voxt::PolishEngine &engine, int port) {
    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) throw std::runtime_error("socket() failed");
    int opt = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = ::inet_addr("127.0.0.1");
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (::bind(srv, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        ::close(srv);
        throw std::runtime_error("bind() failed on port " + std::to_string(port));
    }
    if (::listen(srv, 16) < 0) {
        ::close(srv);
        throw std::runtime_error("listen() failed");
    }
    std::fprintf(stderr, "[voxt_polish_cpp] listening on http://127.0.0.1:%d (pinned_sys_toks=%d, dict_rules=%zu)\n",
                 port, engine.system_prompt_tokens(), engine.alias_rule_count());
    std::fflush(stderr);

    while (true) {
        int cli = ::accept(srv, nullptr, nullptr);
        if (cli < 0) continue;

        // Read HTTP request (headers + Content-Length body)
        std::string req;
        char buf[8192];
        size_t header_end = std::string::npos;
        size_t content_len = 0;
        while (true) {
            ssize_t n = ::recv(cli, buf, sizeof(buf), 0);
            if (n <= 0) break;
            req.append(buf, static_cast<size_t>(n));
            if (header_end == std::string::npos) {
                header_end = req.find("\r\n\r\n");
                if (header_end != std::string::npos) {
                    std::string hdrs = req.substr(0, header_end);
                    for (char &c : hdrs) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
                    size_t cl = hdrs.find("content-length:");
                    if (cl != std::string::npos) {
                        content_len = static_cast<size_t>(std::strtoull(hdrs.c_str() + cl + 15, nullptr, 10));
                    }
                }
            }
            if (header_end != std::string::npos && req.size() >= header_end + 4 + content_len) {
                break;
            }
        }

        if (header_end == std::string::npos) {
            ::close(cli);
            continue;
        }
        const std::string first_line = req.substr(0, req.find("\r\n"));
        const std::string body = req.substr(header_end + 4, content_len);

        if (first_line.rfind("GET /health", 0) == 0) {
            const std::string payload = "{\"status\":\"ok\",\"engine\":\"voxt_polish_cpp\"}";
            const std::string resp =
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                std::to_string(payload.size()) + "\r\nConnection: close\r\n\r\n" + payload;
            send_all(cli, resp);
            ::close(cli);
            continue;
        }

        if (first_line.rfind("POST /v1/chat/completions", 0) == 0) {
            std::string user_text = extract_last_user_content(body);
            bool utter_xml = unwrap_utter_input(user_text);
            bool want_stream = (body.find("\"stream\":true") != std::string::npos ||
                                body.find("\"stream\": true") != std::string::npos);

            auto m = engine.polish(user_text);
            std::string content = utter_xml ? ("<output>" + m.output + "</output>") : m.output;

            if (want_stream) {
                std::ostringstream sse;
                sse << "data: {\"id\":\"voxt-cpp\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\""
                    << json_escape(content)
                    << "\"},\"finish_reason\":null}]}\n\n"
                    << "data: {\"id\":\"voxt-cpp\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":"
                    << m.prompt_tokens << ",\"completion_tokens\":" << m.generated_tokens
                    << ",\"prompt_tokens_details\":{\"cached_tokens\":" << m.kv_reused_tokens << "}}}\n\n"
                    << "data: [DONE]\n\n";
                const std::string sse_str = sse.str();
                const std::string resp =
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nContent-Length: " +
                    std::to_string(sse_str.size()) + "\r\nConnection: close\r\n\r\n" + sse_str;
                send_all(cli, resp);
            } else {
                std::ostringstream js;
                js << "{\"id\":\"voxt-cpp\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\""
                   << json_escape(content)
                   << "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":"
                   << m.prompt_tokens << ",\"completion_tokens\":" << m.generated_tokens
                   << ",\"prompt_tokens_details\":{\"cached_tokens\":" << m.kv_reused_tokens
                   << "}},\"voxt_metrics\":{\"prefill_ms\":" << m.prefill_ms
                   << ",\"decode_ms\":" << m.decode_ms << ",\"total_ms\":" << m.total_ms
                   << ",\"gen_tok_s\":" << m.gen_tok_s
                   << ",\"draft_proposed\":" << m.draft_proposed
                   << ",\"draft_accepted\":" << m.draft_accepted << "}}";
                const std::string payload = js.str();
                const std::string resp =
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                    std::to_string(payload.size()) + "\r\nConnection: close\r\n\r\n" + payload;
                send_all(cli, resp);
            }
            ::close(cli);
            continue;
        }

        const std::string not_found = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        send_all(cli, not_found);
        ::close(cli);
    }
}

void run_bench_jsonl(voxt::PolishEngine &engine, const std::string &jsonl_path, const std::string &out_jsonl) {
    std::ifstream ifs(jsonl_path);
    if (!ifs) throw std::runtime_error("Cannot open jsonl: " + jsonl_path);
    std::vector<std::string> asr_rows;
    std::string line;
    while (std::getline(ifs, line)) {
        std::string asr;
        if (extract_json_string(line, "asr", asr)) {
            asr_rows.push_back(asr);
        }
    }
    if (asr_rows.empty()) throw std::runtime_error("No asr rows found");

    // Warmup 2 calls
    engine.polish(asr_rows[0]);
    engine.polish(asr_rows[0]);

    std::ofstream ofs;
    if (!out_jsonl.empty()) ofs.open(out_jsonl);

    std::vector<double> totals, prefills, decodes, tok_s, accs;
    int64_t tot_gen = 0, tot_steps = 0, tot_prop = 0, tot_acc = 0, tot_reused = 0, tot_prompt = 0;
    int fallbacks = 0;

    for (size_t i = 0; i < asr_rows.size(); ++i) {
        auto m = engine.polish(asr_rows[i]);
        totals.push_back(m.total_ms);
        prefills.push_back(m.prefill_ms);
        decodes.push_back(m.decode_ms);
        if (m.gen_tok_s > 0) tok_s.push_back(m.gen_tok_s);
        if (m.draft_proposed > 0) accs.push_back(static_cast<double>(m.draft_accepted) / m.draft_proposed);
        tot_gen += m.generated_tokens;
        tot_steps += m.decode_steps;
        tot_prop += m.draft_proposed;
        tot_acc += m.draft_accepted;
        tot_reused += m.kv_reused_tokens;
        tot_prompt += m.prompt_tokens;
        if (m.fell_back) ++fallbacks;

        if (ofs.is_open()) {
            ofs << "{\"idx\":" << i
                << ",\"pred\":\"" << json_escape(m.output) << "\""
                << ",\"raw_pred\":\"" << json_escape(m.raw_output) << "\""
                << ",\"total_ms\":" << m.total_ms
                << ",\"prefill_ms\":" << m.prefill_ms
                << ",\"decode_ms\":" << m.decode_ms
                << ",\"gen_tok_s\":" << m.gen_tok_s
                << ",\"kv_reused\":" << m.kv_reused_tokens
                << ",\"kv_prefill\":" << m.kv_prefill_tokens
                << ",\"draft_proposed\":" << m.draft_proposed
                << ",\"draft_accepted\":" << m.draft_accepted
                << ",\"fell_back\":" << (m.fell_back ? "true" : "false")
                << "}\n";
        }
    }

    const double mean_tok_step = tot_steps > 0 ? static_cast<double>(tot_gen) / tot_steps : 0.0;
    const double overall_acc = tot_prop > 0 ? 100.0 * static_cast<double>(tot_acc) / tot_prop : 0.0;
    const double kv_reuse_pct = tot_prompt > 0 ? 100.0 * static_cast<double>(tot_reused) / tot_prompt : 0.0;

    std::printf(
        "{\"rows\":%zu,\"med_total_ms\":%.2f,\"p90_total_ms\":%.2f,"
        "\"med_prefill_ms\":%.2f,\"med_decode_ms\":%.2f,\"med_tok_s\":%.1f,"
        "\"draft_acc_med_pct\":%.1f,\"draft_acc_all_pct\":%.1f,\"tok_per_step\":%.2f,"
        "\"kv_reuse_pct\":%.1f,\"fallbacks\":%d}\n",
        asr_rows.size(), median(totals), percentile(totals, 0.90),
        median(prefills), median(decodes), median(tok_s),
        100.0 * median(accs), overall_acc, mean_tok_step,
        kv_reuse_pct, fallbacks);
}

// Simulate Utter's 12-step incremental growing live-polish session to measure LCP KV reuse gain.
void run_bench_streaming(voxt::PolishEngine &engine) {
    const std::vector<std::string> steps = {
        "我再试一下",
        "我再试一下我的这个",
        "我再试一下我的这个语音输入",
        "我再试一下我的这个语音输入，看有没有",
        "我再试一下我的这个语音输入，看有没有什么问题。",
        "我再试一下我的这个语音输入，看有没有什么问题。我还是觉得",
        "我再试一下我的这个语音输入，看有没有什么问题。我还是觉得它第一个 token",
        "我再试一下我的这个语音输入，看有没有什么问题。我还是觉得它第一个 token 的速度有点慢，",
        "我再试一下我的这个语音输入，看有没有什么问题。我还是觉得它第一个 token 的速度有点慢，整个部分",
        "我再试一下我的这个语音输入，看有没有什么问题。我还是觉得它第一个 token 的速度有点慢，整个部分怎么样加速。",
        "我再试一下我的这个语音输入，看有没有什么问题。我还是觉得它第一个 token 的速度有点慢，整个部分怎么样加速。你有什么想法吗？",
        "我再试一下我的这个语音输入，看有没有什么问题。我还是觉得它第一个 token 的速度有点慢，整个部分怎么样加速。你有什么想法吗？你可以给我讲一下你的建议。"
    };
    engine.polish(steps[0]);

    auto run_mode = [&](bool lcp_on, const char *label) {
        engine.set_lcp_reuse(lcp_on);
        engine.reset_to_system_prefix();
        double sum_prefill = 0, sum_total = 0;
        int64_t sum_reused = 0, sum_new = 0, sum_prompt = 0;
        for (const auto &s : steps) {
            auto m = engine.polish(s);
            sum_prefill += m.prefill_ms;
            sum_total += m.total_ms;
            sum_reused += m.kv_reused_tokens;
            sum_new += m.kv_prefill_tokens;
            sum_prompt += m.prompt_tokens;
        }
        std::printf(
            "{\"mode\":\"%s\",\"steps\":%zu,\"sum_prefill_ms\":%.2f,\"sum_total_ms\":%.2f,"
            "\"kv_reused_tokens\":%lld,\"kv_new_prefill_tokens\":%lld,\"kv_reuse_pct\":%.1f}\n",
            label, steps.size(), sum_prefill, sum_total,
            static_cast<long long>(sum_reused), static_cast<long long>(sum_new),
            100.0 * static_cast<double>(sum_reused) / static_cast<double>(sum_prompt));
    };

    run_mode(false, "pinned_sys_only");
    run_mode(true, "pinned_sys_plus_incremental_lcp");
}

} // namespace

int main(int argc, char **argv) {
    voxt::EngineConfig cfg;
    std::string prompt_path;
    std::string bench_jsonl;
    std::string out_jsonl;
    bool bench_streaming = false;
    bool serve = false;
    int port = 8093;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--model" && i + 1 < argc) cfg.model_path = argv[++i];
        else if (a == "--system-prompt" && i + 1 < argc) prompt_path = argv[++i];
        else if (a == "--dict" && i + 1 < argc) cfg.dict_path = argv[++i];
        else if (a == "--draft-max" && i + 1 < argc) cfg.draft_max = std::stoi(argv[++i]);
        else if (a == "--no-spec") cfg.enable_spec = false;
        else if (a == "--fa-on") cfg.flash_attn = true;
        else if (a == "--no-post") cfg.enable_postprocess = false;
        else if (a == "--no-guardrail") cfg.enable_guardrail = false;
        else if (a == "--bench-jsonl" && i + 1 < argc) bench_jsonl = argv[++i];
        else if (a == "--out-jsonl" && i + 1 < argc) out_jsonl = argv[++i];
        else if (a == "--bench-streaming") bench_streaming = true;
        else if (a == "--serve") serve = true;
        else if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
    }

    if (cfg.model_path.empty() || prompt_path.empty()) {
        std::fprintf(stderr,
            "Usage: %s --model <gguf> --system-prompt <txt> [--dict <json>] "
            "[--no-spec] [--draft-max 6] [--bench-jsonl <in> --out-jsonl <out>] "
            "[--bench-streaming] [--serve --port 8093]\n", argv[0]);
        return 1;
    }

    try {
        cfg.system_prompt = read_file(prompt_path);
        voxt::PolishEngine engine(cfg);
        if (!bench_jsonl.empty()) {
            run_bench_jsonl(engine, bench_jsonl, out_jsonl);
        } else if (bench_streaming) {
            run_bench_streaming(engine);
        } else if (serve) {
            run_http_server(engine, port);
        } else {
            auto m = engine.polish("嗯那个我想测试一下就是说 C++ 直接链接 libllama 的速度。");
            std::printf("out=%s | prefill=%.1fms decode=%.1fms total=%.1fms tok/s=%.1f\n",
                        m.output.c_str(), m.prefill_ms, m.decode_ms, m.total_ms, m.gen_tok_s);
        }
    } catch (const std::exception &e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
    return 0;
}
