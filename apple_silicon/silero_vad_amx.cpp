// Pure C++ Apple Accelerate (AMX / vDSP / cblas) Silero VAD v6 engine.
// Zero Metal GPU usage, zero CoreML IPC overhead, reads Voxt's exact MLX
// safetensors weights directly. Supports both streaming (chunk=1, 32ms) and
// offline batched modes.
//
// Build:
//   clang++ -O3 -std=c++17 -framework Accelerate silero_vad_amx.cpp -o silero_vad_amx

#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kSampleRate = 16000;
constexpr int kChunkSamples = 512;
constexpr int kContextSamples = 64;
constexpr int kWindowSamples = kContextSamples + kChunkSamples; // 576
constexpr int kPadSamples = 64;
constexpr int kPaddedSamples = kWindowSamples + kPadSamples;    // 640
constexpr int kStftKernel = 256;
constexpr int kStftHop = 128;
constexpr int kStftFrames = 4;                                  // (640-256)/128 + 1
constexpr int kStftBins = 129;                                  // 258 / 2
constexpr int kHidden = 128;

struct TensorView {
  std::vector<int64_t> shape;
  size_t offset = 0;
  size_t nbytes = 0;
};

// Minimal safetensors parser (extracts float32 tensors with "vad_16k." prefix).
struct SafetensorsFile {
  std::vector<uint8_t> data;
  size_t payload_offset = 0;
  std::unordered_map<std::string, TensorView> tensors;

  explicit SafetensorsFile(const std::string &path) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs) {
      throw std::runtime_error("Cannot open safetensors: " + path);
    }
    const size_t size = static_cast<size_t>(ifs.tellg());
    ifs.seekg(0);
    data.resize(size);
    ifs.read(reinterpret_cast<char *>(data.data()), size);
    if (size < 8) {
      throw std::runtime_error("Truncated safetensors");
    }
    uint64_t header_len = 0;
    std::memcpy(&header_len, data.data(), 8);
    payload_offset = 8 + static_cast<size_t>(header_len);
    std::string header(reinterpret_cast<const char *>(data.data() + 8),
                       static_cast<size_t>(header_len));
    ParseHeader(header);
  }

  const float *GetFloat(const std::string &name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) {
      throw std::runtime_error("Missing tensor: " + name);
    }
    return reinterpret_cast<const float *>(data.data() + payload_offset +
                                           it->second.offset);
  }

 private:
  static size_t FindMatchingBrace(const std::string &s, size_t open_pos) {
    int depth = 0;
    bool in_str = false;
    for (size_t i = open_pos; i < s.size(); ++i) {
      if (s[i] == '"' && (i == 0 || s[i - 1] != '\\')) {
        in_str = !in_str;
      } else if (!in_str) {
        if (s[i] == '{') ++depth;
        else if (s[i] == '}') {
          if (--depth == 0) return i;
        }
      }
    }
    return std::string::npos;
  }

  void ParseHeader(const std::string &h) {
    size_t pos = h.find('{');
    if (pos == std::string::npos) return;
    ++pos;
    while (pos < h.size()) {
      size_t q1 = h.find('"', pos);
      if (q1 == std::string::npos) break;
      size_t q2 = h.find('"', q1 + 1);
      if (q2 == std::string::npos) break;
      std::string key = h.substr(q1 + 1, q2 - q1 - 1);
      size_t obj_start = h.find('{', q2 + 1);
      if (obj_start == std::string::npos) break;
      size_t obj_end = FindMatchingBrace(h, obj_start);
      if (obj_end == std::string::npos) break;
      pos = obj_end + 1;
      if (key == "__metadata__") continue;

      std::string obj = h.substr(obj_start, obj_end - obj_start + 1);
      size_t off_pos = obj.find("\"data_offsets\"");
      if (off_pos == std::string::npos) continue;
      size_t b1 = obj.find('[', off_pos);
      size_t comma = obj.find(',', b1);
      size_t b2 = obj.find(']', comma);
      size_t start_off = std::stoull(obj.substr(b1 + 1, comma - b1 - 1));
      size_t end_off = std::stoull(obj.substr(comma + 1, b2 - comma - 1));

      TensorView tv;
      tv.offset = start_off;
      tv.nbytes = end_off - start_off;
      tensors.emplace(std::move(key), std::move(tv));
    }
  }
};

struct WavData {
  std::vector<float> samples;
};

WavData LoadWavPCM16(const std::string &path) {
  std::ifstream ifs(path, std::ios::binary | std::ios::ate);
  if (!ifs) throw std::runtime_error("Cannot open WAV: " + path);
  const size_t size = static_cast<size_t>(ifs.tellg());
  ifs.seekg(0);
  std::vector<uint8_t> buf(size);
  ifs.read(reinterpret_cast<char *>(buf.data()), size);
  size_t off = 12;
  while (off + 8 <= size) {
    std::string id(reinterpret_cast<const char *>(buf.data() + off), 4);
    uint32_t sz = 0;
    std::memcpy(&sz, buf.data() + off + 4, 4);
    off += 8;
    if (id == "data") {
      const size_t end = std::min(size, off + static_cast<size_t>(sz));
      const size_t n = (end - off) / 2;
      WavData out;
      out.samples.resize(n);
      const int16_t *pcm = reinterpret_cast<const int16_t *>(buf.data() + off);
      constexpr float scale = 1.0f / 32768.0f;
      for (size_t i = 0; i < n; ++i) {
        out.samples[i] = static_cast<float>(pcm[i]) * scale;
      }
      return out;
    }
    off += sz + (sz & 1u);
  }
  throw std::runtime_error("No data chunk in WAV: " + path);
}

struct StreamState {
  float h[kHidden] = {};
  float c[kHidden] = {};
  float ctx[kContextSamples] = {};

  void Reset() {
    std::memset(h, 0, sizeof(h));
    std::memset(c, 0, sizeof(c));
    std::memset(ctx, 0, sizeof(ctx));
  }
};

class SileroVadAmx {
 public:
  explicit SileroVadAmx(const std::string &safetensors_path)
      : sf_(safetensors_path) {
    stft_w_ = sf_.GetFloat("vad_16k.stft_conv.weight"); // [258, 256]
    conv1_w_ = sf_.GetFloat("vad_16k.conv1.weight");    // [128, 3*129]
    conv1_b_ = sf_.GetFloat("vad_16k.conv1.bias");      // [128]
    conv2_w_ = sf_.GetFloat("vad_16k.conv2.weight");    // [64, 3*128]
    conv2_b_ = sf_.GetFloat("vad_16k.conv2.bias");      // [64]
    conv3_w_ = sf_.GetFloat("vad_16k.conv3.weight");    // [64, 3*64]
    conv3_b_ = sf_.GetFloat("vad_16k.conv3.bias");      // [64]
    conv4_w_ = sf_.GetFloat("vad_16k.conv4.weight");    // [128, 3*64]
    conv4_b_ = sf_.GetFloat("vad_16k.conv4.bias");      // [128]
    lstm_wx_ = sf_.GetFloat("vad_16k.lstm.Wx");         // [512, 128]
    lstm_wh_ = sf_.GetFloat("vad_16k.lstm.Wh");         // [512, 128]
    lstm_b_ = sf_.GetFloat("vad_16k.lstm.bias");        // [512]
    final_w_ = sf_.GetFloat("vad_16k.final_conv.weight"); // [128]
    final_b_ = sf_.GetFloat("vad_16k.final_conv.bias");   // [1]
  }

  // Feed `chunk_count` consecutive 512-sample chunks (works for chunk_count=1
  // streaming OR chunk_count=N offline batched).
  void Feed(const float *samples, int chunk_count, StreamState &st,
            float *out_probs) const {
    if (chunk_count <= 0) return;

    // 1. Build padded windows + extract 4 STFT frames of 256 samples per chunk:
    //    stft_in: [chunk_count * 4, 256]
    const int n_stft_rows = chunk_count * kStftFrames;
    stft_in_.resize(static_cast<size_t>(n_stft_rows) * kStftKernel);
    float win[kPaddedSamples];
    for (int ch = 0; ch < chunk_count; ++ch) {
      const float *ctx_src =
          (ch == 0) ? st.ctx : (samples + ch * kChunkSamples - kContextSamples);
      std::memcpy(win, ctx_src, kContextSamples * sizeof(float));
      std::memcpy(win + kContextSamples, samples + ch * kChunkSamples,
                  kChunkSamples * sizeof(float));
      for (int i = 0; i < kPadSamples; ++i) {
        win[kWindowSamples + i] = win[kWindowSamples - 2 - i];
      }
      for (int f = 0; f < kStftFrames; ++f) {
        std::memcpy(stft_in_.data() +
                        static_cast<size_t>(ch * kStftFrames + f) * kStftKernel,
                    win + f * kStftHop, kStftKernel * sizeof(float));
      }
    }
    const float *last_end = samples + chunk_count * kChunkSamples;
    std::memcpy(st.ctx, last_end - kContextSamples,
                kContextSamples * sizeof(float));

    // STFT GEMM: [n_stft_rows, 256] * [258, 256]^T -> [n_stft_rows, 258]
    stft_out_.resize(static_cast<size_t>(n_stft_rows) * (kStftBins * 2));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n_stft_rows, kStftBins * 2, kStftKernel,
                1.0f, stft_in_.data(), kStftKernel,
                stft_w_, kStftKernel,
                0.0f, stft_out_.data(), kStftBins * 2);

    // Magnitude: [chunk_count, 4, 129]
    mag_.resize(static_cast<size_t>(n_stft_rows) * kStftBins);
    for (int r = 0; r < n_stft_rows; ++r) {
      const float *re = stft_out_.data() + static_cast<size_t>(r) * (kStftBins * 2);
      const float *im = re + kStftBins;
      float *dst = mag_.data() + static_cast<size_t>(r) * kStftBins;
      // dst = sqrt(re^2 + im^2)
      vDSP_vsq(re, 1, tmp129_, 1, kStftBins);
      vDSP_vma(im, 1, im, 1, tmp129_, 1, dst, 1, kStftBins);
      int n = kStftBins;
      vvsqrtf(dst, dst, &n);
    }

    // 2. Conv1..4 with im2col + cblas_sgemm + ReLU
    // conv1: [B, 4, 129] -> [B, 4, 128] (k=3, s=1, p=1)
    Conv1dRelu(mag_.data(), chunk_count, 4, 129, 128, 1, conv1_w_, conv1_b_, c1_);
    // conv2: [B, 4, 128] -> [B, 2, 64] (k=3, s=2, p=1)
    Conv1dRelu(c1_.data(), chunk_count, 4, 128, 64, 2, conv2_w_, conv2_b_, c2_);
    // conv3: [B, 2, 64] -> [B, 1, 64] (k=3, s=2, p=1)
    Conv1dRelu(c2_.data(), chunk_count, 2, 64, 64, 2, conv3_w_, conv3_b_, c3_);
    // conv4: [B, 1, 64] -> [B, 1, 128] (k=3, s=1, p=1)
    Conv1dRelu(c3_.data(), chunk_count, 1, 64, 128, 1, conv4_w_, conv4_b_, c4_);

    // 3. LSTM input gates for all chunks in ONE GEMM:
    //    x_gates[B, 512] = c4[B, 128] * Wx[512, 128]^T + bias[512]
    x_gates_.resize(static_cast<size_t>(chunk_count) * 512);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                chunk_count, 512, kHidden,
                1.0f, c4_.data(), kHidden,
                lstm_wx_, kHidden,
                0.0f, x_gates_.data(), 512);

    // 4. Recurrent LSTM loop (chunk by chunk) + final_conv + sigmoid
    float gates[512];
    float h_relu[kHidden];
    for (int ch = 0; ch < chunk_count; ++ch) {
      const float *xg = x_gates_.data() + static_cast<size_t>(ch) * 512;
      // gates = Wh * h
      cblas_sgemv(CblasRowMajor, CblasNoTrans, 512, kHidden,
                  1.0f, lstm_wh_, kHidden, st.h, 1,
                  0.0f, gates, 1);
      // gates += xg + lstm_b
      vDSP_vadd(gates, 1, xg, 1, gates, 1, 512);
      vDSP_vadd(gates, 1, lstm_b_, 1, gates, 1, 512);

      for (int i = 0; i < kHidden; ++i) {
        const float ig = 1.0f / (1.0f + std::exp(-gates[i]));
        const float fg = 1.0f / (1.0f + std::exp(-gates[kHidden + i]));
        const float gg = std::tanh(gates[2 * kHidden + i]);
        const float og = 1.0f / (1.0f + std::exp(-gates[3 * kHidden + i]));
        const float nc = fg * st.c[i] + ig * gg;
        const float nh = og * std::tanh(nc);
        st.c[i] = nc;
        st.h[i] = nh;
        h_relu[i] = nh > 0.0f ? nh : 0.0f;
      }

      float logit = final_b_[0];
      vDSP_dotpr(h_relu, 1, final_w_, 1, &logit, kHidden);
      logit += final_b_[0];
      out_probs[ch] = 1.0f / (1.0f + std::exp(-logit));
    }
  }

 private:
  void Conv1dRelu(const float *in, int B, int T_in, int C_in, int C_out,
                  int stride, const float *w, const float *b,
                  std::vector<float> &out) const {
    const int T_out = (T_in + 2 - 3) / stride + 1;
    const int N_rows = B * T_out;
    const int K = 3 * C_in;
    im2col_.assign(static_cast<size_t>(N_rows) * K, 0.0f);

    for (int batch = 0; batch < B; ++batch) {
      const float *in_b = in + static_cast<size_t>(batch) * T_in * C_in;
      for (int to = 0; to < T_out; ++to) {
        float *row = im2col_.data() + static_cast<size_t>(batch * T_out + to) * K;
        const int ti_center = to * stride;
        for (int k = 0; k < 3; ++k) {
          const int ti = ti_center + k - 1;
          if (ti >= 0 && ti < T_in) {
            std::memcpy(row + k * C_in, in_b + ti * C_in, C_in * sizeof(float));
          }
        }
      }
    }

    out.resize(static_cast<size_t>(N_rows) * C_out);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N_rows, C_out, K,
                1.0f, im2col_.data(), K,
                w, K,
                0.0f, out.data(), C_out);

    for (int r = 0; r < N_rows; ++r) {
      float *y = out.data() + static_cast<size_t>(r) * C_out;
      vDSP_vadd(y, 1, b, 1, y, 1, C_out);
      const float zero = 0.0f;
      vDSP_vthres(y, 1, &zero, y, 1, C_out);
    }
  }

  SafetensorsFile sf_;
  const float *stft_w_ = nullptr;
  const float *conv1_w_ = nullptr, *conv1_b_ = nullptr;
  const float *conv2_w_ = nullptr, *conv2_b_ = nullptr;
  const float *conv3_w_ = nullptr, *conv3_b_ = nullptr;
  const float *conv4_w_ = nullptr, *conv4_b_ = nullptr;
  const float *lstm_wx_ = nullptr, *lstm_wh_ = nullptr, *lstm_b_ = nullptr;
  const float *final_w_ = nullptr, *final_b_ = nullptr;

  mutable std::vector<float> stft_in_, stft_out_, mag_;
  mutable std::vector<float> im2col_, c1_, c2_, c3_, c4_, x_gates_;
  mutable float tmp129_[kStftBins];
};

std::string Basename(const std::string &path) {
  auto pos = path.find_last_of('/');
  return pos == std::string::npos ? path : path.substr(pos + 1);
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cerr << "Usage: silero_vad_amx <model.safetensors> <stream|batch|b32> <wav1> ...\n";
    return 1;
  }
  const std::string model_path = argv[1];
  const std::string mode = argv[2];
  SileroVadAmx vad(model_path);

  // Warmup
  {
    std::vector<float> zeros(kChunkSamples * 32, 0.0f);
    std::vector<float> probs(32);
    StreamState st;
    vad.Feed(zeros.data(), mode == "stream" ? 1 : 32, st, probs.data());
  }

  for (int argi = 3; argi < argc; ++argi) {
    const std::string wav_path = argv[argi];
    WavData wav = LoadWavPCM16(wav_path);
    const int n_chunks =
        static_cast<int>((wav.samples.size() + kChunkSamples - 1) / kChunkSamples);
    wav.samples.resize(static_cast<size_t>(n_chunks) * kChunkSamples, 0.0f);
    std::vector<float> probs(n_chunks, 0.0f);
    StreamState st;

    const auto t0 = std::chrono::steady_clock::now();
    if (mode == "stream") {
      for (int ch = 0; ch < n_chunks; ++ch) {
        vad.Feed(wav.samples.data() + ch * kChunkSamples, 1, st, probs.data() + ch);
      }
    } else if (mode == "b32") {
      for (int ch = 0; ch < n_chunks; ch += 32) {
        const int take = std::min(32, n_chunks - ch);
        vad.Feed(wav.samples.data() + ch * kChunkSamples, take, st, probs.data() + ch);
      }
    } else {
      vad.Feed(wav.samples.data(), n_chunks, st, probs.data());
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double audio_sec = static_cast<double>(wav.samples.size()) / kSampleRate;
    int speech_chunks = 0;
    for (float p : probs) {
      if (p >= 0.5f) ++speech_chunks;
    }

    std::cout << std::fixed
              << "{\"backend\":\"accelerate_amx_" << mode << "\""
              << ",\"wav_file\":\"" << Basename(wav_path) << "\""
              << ",\"chunks\":" << n_chunks
              << ",\"audio_sec\":" << std::setprecision(2) << audio_sec
              << ",\"total_ms\":" << std::setprecision(3) << ms
              << ",\"per_chunk_us\":" << std::setprecision(2) << (ms * 1000.0 / n_chunks)
              << ",\"rtf\":" << std::setprecision(6) << (ms / 1000.0 / audio_sec)
              << ",\"speech_chunks\":" << speech_chunks
              << ",\"first5\":[" << std::setprecision(4)
              << probs[0] << "," << probs[1] << "," << probs[2] << ","
              << probs[3] << "," << probs[4] << "]}\n";
  }
  return 0;
}
