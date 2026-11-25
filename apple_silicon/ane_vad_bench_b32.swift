import CoreML
import Foundation

struct BenchSummary: Codable {
    let compute_unit: String
    let batch_chunks: Int
    let wav_file: String
    let chunks: Int
    let calls: Int
    let audio_sec: Double
    let total_ms: Double
    let per_chunk_us: Double
    let rtf: Double
    let speech_chunks: Int
}

func loadWavPCM16(_ path: String) throws -> [Float] {
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
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
    throw NSError(domain: "WAV", code: 1)
}

func parseCU(_ name: String) -> MLComputeUnits {
    switch name {
    case "cpu_only": return .cpuOnly
    case "cpu_and_gpu": return .cpuAndGPU
    case "cpu_and_ne": return .cpuAndNeuralEngine
    default: return .all
    }
}

let args = CommandLine.arguments
if args.count < 4 {
    fputs("Usage: ane_vad_bench_b32 <mlmodelc> <cpu_and_ne|cpu_only|cpu_and_gpu> <wav1> ...\n", stderr)
    exit(1)
}

let B = 32
let cfg = MLModelConfiguration()
cfg.computeUnits = parseCU(args[2])
let model = try MLModel(contentsOf: URL(fileURLWithPath: args[1]), configuration: cfg)

let wArr = try MLMultiArray(shape: [NSNumber(value: B), 576], dataType: .float32)
let hArr = try MLMultiArray(shape: [1, 128], dataType: .float32)
let cArr = try MLMultiArray(shape: [1, 128], dataType: .float32)

func zeroState() {
    let hp = hArr.dataPointer.bindMemory(to: Float.self, capacity: 128)
    let cp = cArr.dataPointer.bindMemory(to: Float.self, capacity: 128)
    for i in 0..<128 { hp[i] = 0; cp[i] = 0 }
}

// Warmup 3 batched calls
zeroState()
for _ in 0..<3 {
    let inp = try MLDictionaryFeatureProvider(dictionary: [
        "windows": MLFeatureValue(multiArray: wArr),
        "h_in": MLFeatureValue(multiArray: hArr),
        "c_in": MLFeatureValue(multiArray: cArr),
    ])
    _ = try model.prediction(from: inp)
}

let encoder = JSONEncoder()
for wavPath in args[3...] {
    let samples = try loadWavPCM16(wavPath)
    let nChunks = (samples.count + 511) / 512
    let nCalls = (nChunks + B - 1) / B
    var padded = samples
    let needSamples = nCalls * B * 512
    if padded.count < needSamples {
        padded.append(contentsOf: repeatElement(Float(0), count: needSamples - padded.count))
    }

    zeroState()
    var ctx = [Float](repeating: 0, count: 64)
    var probs = [Float]()
    probs.reserveCapacity(nChunks)

    let wp = wArr.dataPointer.bindMemory(to: Float.self, capacity: B * 576)
    let hp = hArr.dataPointer.bindMemory(to: Float.self, capacity: 128)
    let cp = cArr.dataPointer.bindMemory(to: Float.self, capacity: 128)

    let t0 = DispatchTime.now().uptimeNanoseconds
    for call in 0..<nCalls {
        for b in 0..<B {
            let ch = call * B + b
            let rowOff = b * 576
            for i in 0..<64 { wp[rowOff + i] = ctx[i] }
            let base = ch * 512
            for i in 0..<512 { wp[rowOff + 64 + i] = padded[base + i] }
            for i in 0..<64 { ctx[i] = padded[base + 448 + i] }
        }
        let inp = try MLDictionaryFeatureProvider(dictionary: [
            "windows": MLFeatureValue(multiArray: wArr),
            "h_in": MLFeatureValue(multiArray: hArr),
            "c_in": MLFeatureValue(multiArray: cArr),
        ])
        let out = try model.prediction(from: inp)
        let pArr = out.featureValue(for: "probs")!.multiArrayValue!
        let hOut = out.featureValue(for: "h_out")!.multiArrayValue!
        let cOut = out.featureValue(for: "c_out")!.multiArrayValue!
        let valid = min(B, nChunks - call * B)
        for b in 0..<valid { probs.append(pArr[b].floatValue) }
        for i in 0..<128 {
            hp[i] = hOut[i].floatValue
            cp[i] = cOut[i].floatValue
        }
    }
    let elapsedMs = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000.0
    let audioSec = Double(samples.count) / 16000.0
    let s = BenchSummary(
        compute_unit: args[2],
        batch_chunks: B,
        wav_file: URL(fileURLWithPath: wavPath).lastPathComponent,
        chunks: nChunks,
        calls: nCalls,
        audio_sec: (audioSec * 100).rounded() / 100,
        total_ms: (elapsedMs * 100).rounded() / 100,
        per_chunk_us: ((elapsedMs * 1000.0 / Double(nChunks)) * 10).rounded() / 10,
        rtf: ((elapsedMs / 1000.0 / audioSec) * 10000).rounded() / 10000,
        speech_chunks: probs.filter { $0 >= 0.5 }.count
    )
    print(String(data: try encoder.encode(s), encoding: .utf8)!)
}
