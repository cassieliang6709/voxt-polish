import CoreML
import Foundation

struct BenchSummary: Codable {
    let compute_unit: String
    let wav_file: String
    let chunks: Int
    let audio_sec: Double
    let total_ms: Double
    let per_chunk_us: Double
    let rtf: Double
    let speech_chunks: Int
    let probs_sample: [Double]
}

func loadWavPCM16(_ path: String) throws -> [Float] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    // Find "data" chunk in WAV header
    var offset = 12
    while offset + 8 <= data.count {
        let id = String(data: data[offset..<(offset + 4)], encoding: .ascii) ?? ""
        let size = data.withUnsafeBytes { raw -> UInt32 in
            raw.loadUnaligned(fromByteOffset: offset + 4, as: UInt32.self)
        }
        offset += 8
        if id == "data" {
            let end = min(data.count, offset + Int(size))
            let nSamples = (end - offset) / 2
            var out = [Float](repeating: 0, count: nSamples)
            data.withUnsafeBytes { raw in
                for i in 0..<nSamples {
                    let s = raw.loadUnaligned(fromByteOffset: offset + i * 2, as: Int16.self)
                    out[i] = Float(s) / 32768.0
                }
            }
            return out
        }
        offset += Int(size)
    }
    throw NSError(domain: "WAV", code: 1, userInfo: [NSLocalizedDescriptionKey: "No data chunk"])
}

func parseComputeUnit(_ name: String) -> MLComputeUnits {
    switch name {
    case "cpu_only": return .cpuOnly
    case "cpu_and_gpu": return .cpuAndGPU
    case "cpu_and_ne": return .cpuAndNeuralEngine
    default: return .all
    }
}

let args = CommandLine.arguments
if args.count < 4 {
    fputs("Usage: ane_vad_bench <mlmodelc_or_mlpackage> <cpu_and_ne|cpu_only|cpu_and_gpu|all> <wav1> [wav2...]\n", stderr)
    exit(1)
}

let modelURL = URL(fileURLWithPath: args[1])
let cuName = args[2]
let wavPaths = Array(args[3...])

let compiledURL: URL
if modelURL.pathExtension == "mlmodelc" {
    compiledURL = modelURL
} else {
    compiledURL = try MLModel.compileModel(at: modelURL)
}

let cfg = MLModelConfiguration()
cfg.computeUnits = parseComputeUnit(cuName)
let model = try MLModel(contentsOf: compiledURL, configuration: cfg)

let wArr = try MLMultiArray(shape: [1, 576], dataType: .float32)
let hArr = try MLMultiArray(shape: [1, 128], dataType: .float32)
let cArr = try MLMultiArray(shape: [1, 128], dataType: .float32)

func zeroState() {
    let hp = hArr.dataPointer.bindMemory(to: Float.self, capacity: 128)
    let cp = cArr.dataPointer.bindMemory(to: Float.self, capacity: 128)
    for i in 0..<128 { hp[i] = 0; cp[i] = 0 }
}

// Warmup 10 chunks
zeroState()
for _ in 0..<10 {
    let inp = try MLDictionaryFeatureProvider(dictionary: [
        "audio_window": MLFeatureValue(multiArray: wArr),
        "h_in": MLFeatureValue(multiArray: hArr),
        "c_in": MLFeatureValue(multiArray: cArr),
    ])
    _ = try model.prediction(from: inp)
}

let encoder = JSONEncoder()
for wavPath in wavPaths {
    let samples = try loadWavPCM16(wavPath)
    let nChunks = (samples.count + 511) / 512
    var padded = samples
    if padded.count < nChunks * 512 {
        padded.append(contentsOf: repeatElement(Float(0), count: nChunks * 512 - padded.count))
    }

    zeroState()
    var ctx = [Float](repeating: 0, count: 64)
    var probs = [Float]()
    probs.reserveCapacity(nChunks)

    let wp = wArr.dataPointer.bindMemory(to: Float.self, capacity: 576)
    let hp = hArr.dataPointer.bindMemory(to: Float.self, capacity: 128)
    let cp = cArr.dataPointer.bindMemory(to: Float.self, capacity: 128)

    let t0 = DispatchTime.now().uptimeNanoseconds
    for ch in 0..<nChunks {
        for i in 0..<64 { wp[i] = ctx[i] }
        let base = ch * 512
        for i in 0..<512 { wp[64 + i] = padded[base + i] }
        for i in 0..<64 { ctx[i] = padded[base + 448 + i] }

        let inp = try MLDictionaryFeatureProvider(dictionary: [
            "audio_window": MLFeatureValue(multiArray: wArr),
            "h_in": MLFeatureValue(multiArray: hArr),
            "c_in": MLFeatureValue(multiArray: cArr),
        ])
        let out = try model.prediction(from: inp)
        let probArr = out.featureValue(for: "prob")!.multiArrayValue!
        let hOut = out.featureValue(for: "h_out")!.multiArrayValue!
        let cOut = out.featureValue(for: "c_out")!.multiArrayValue!

        probs.append(probArr[0].floatValue)
        for i in 0..<128 {
            hp[i] = hOut[i].floatValue
            cp[i] = cOut[i].floatValue
        }
    }
    let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000.0
    let audioSec = Double(samples.count) / 16000.0
    let speechChunks = probs.filter { $0 >= 0.5 }.count
    let sampleProbs = stride(from: 0, to: probs.count, by: max(1, probs.count / 20)).map { Double(probs[$0]) }
    let summary = BenchSummary(
        compute_unit: cuName,
        wav_file: URL(fileURLWithPath: wavPath).lastPathComponent,
        chunks: nChunks,
        audio_sec: (audioSec * 100).rounded() / 100,
        total_ms: (elapsedMs * 100).rounded() / 100,
        per_chunk_us: ((elapsedMs * 1000.0 / Double(nChunks)) * 10).rounded() / 10,
        rtf: ((elapsedMs / 1000.0 / audioSec) * 10000).rounded() / 10000,
        speech_chunks: speechChunks,
        probs_sample: sampleProbs.map { ($0 * 10000).rounded() / 10000 }
    )
    let j = try encoder.encode(summary)
    print(String(data: j, encoding: .utf8)!)
}
