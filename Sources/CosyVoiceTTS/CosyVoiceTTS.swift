import Foundation
import MLXCommon
import MLX
import MLXNN
import AudioCommon

/// Error types for CosyVoice TTS
public enum CosyVoiceTTSError: Error, LocalizedError {
    case modelLoadFailed(String)
    case downloadFailed(String)
    case invalidInput(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let msg): return "Model load failed: \(msg)"
        case .downloadFailed(let msg): return "Download failed: \(msg)"
        case .invalidInput(let msg): return "Invalid input: \(msg)"
        case .generationFailed(let msg): return "Generation failed: \(msg)"
        }
    }
}

/// CosyVoice3 TTS model — generates speech from text.
///
/// Three-stage pipeline:
/// 1. LLM (Qwen2.5-0.5B) generates speech tokens from text
/// 2. Flow matching (DiT) converts tokens to mel spectrogram
/// 3. HiFi-GAN vocoder converts mel to 24kHz audio waveform
///
/// - Warning: This class is not thread-safe. Create separate instances for concurrent use.
public final class CosyVoiceTTSModel {
    /// Default HuggingFace model identifier — the 0.5B bf16 MLX bundle. Single
    /// SSOT for the registry and other call sites.
    public static let defaultModelId = "aufklarer/CosyVoice3-0.5B-MLX-bf16"

    public let config: CosyVoiceConfig

    let llm: CosyVoiceLLM
    let flow: CosyVoiceFlowModel
    let hifigan: HiFiGANGenerator
    let tokenizer: Qwen3Tokenizer

    /// Whether the model weights are loaded and ready for inference.
    var _isLoaded = true

    /// Initialize with config
    public init(config: CosyVoiceConfig = .default) {
        self.config = config
        self.llm = CosyVoiceLLM(config: config.llm)
        self.flow = CosyVoiceFlowModel(config: config.flow)
        self.hifigan = HiFiGANGenerator(config: config.hifigan)
        self.tokenizer = Qwen3Tokenizer()
    }

    /// Download and load model from HuggingFace
    ///
    /// Downloads three safetensors files: llm.safetensors, flow.safetensors, hifigan.safetensors
    /// Caches to ~/Library/Caches/qwen3-speech/
    public static func fromPretrained(
        modelId: String = CosyVoiceTTSModel.defaultModelId,
        cacheDir: URL? = nil,
        offlineMode: Bool = false,
        progressHandler: ((Double, String) -> Void)? = nil
    ) async throws -> CosyVoiceTTSModel {
        // Get cache directory
        let cacheDir = try cacheDir ?? HuggingFaceDownloader.getCacheDirectory(for: modelId)

        // Download if needed (check both weights and tokenizer)
        let needsWeights = !HuggingFaceDownloader.weightsExist(in: cacheDir)
        let needsTokenizer = !FileManager.default.fileExists(
            atPath: cacheDir.appendingPathComponent("vocab.json").path)

        if needsWeights || needsTokenizer {
            progressHandler?(0.0, "Downloading model files...")
            try await HuggingFaceDownloader.downloadWeights(
                modelId: modelId,
                to: cacheDir,
                additionalFiles: [
                    "llm.safetensors", "flow.safetensors", "hifigan.safetensors",
                    "vocab.json", "merges.txt", "tokenizer_config.json", "config.json",
                ],
                offlineMode: offlineMode
            ) { progress in
                progressHandler?(progress * 0.5, "Downloading...")
            }
        }

        // Read the bundle's `config.json` so the LLM/DiT modules can be told
        // the correct quantization bits. The bf16 bundle omits the
        // `quantization` block entirely; in that case we keep the static
        // defaults and the loader detects bf16 via the absence of `.scales`
        // tensors in the safetensors.
        var config = CosyVoiceConfig.default
        let configURL = cacheDir.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: configURL.path),
           let data = try? Data(contentsOf: configURL),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let quant = json["quantization"] as? [String: Any] {
                // `bits` is the LLM bit width; `llm_bits` is accepted for
                // bundles that also carry per-component metadata.
                if let gs = quant["group_size"] as? Int { config.llm.groupSize = gs }
                if let bits = (quant["llm_bits"] as? Int) ?? (quant["bits"] as? Int) {
                    switch bits {
                    case 8:
                        config.llm.bits = bits
                        print("  Bundle quantization (LLM): \(config.llm.bits)-bit (group_size \(config.llm.groupSize))")
                    case 16:
                        config.llm.bits = bits
                        print("  Bundle precision (LLM): 16-bit (plain Linear; no .scales)")
                    default:
                        throw CosyVoiceTTSError.modelLoadFailed(
                            "CosyVoice LLM bundles must be 8-bit quantized or 16-bit/bf16 plain Linear.")
                    }
                } else {
                    print("  Bundle quantization (LLM): \(config.llm.bits)-bit (group_size \(config.llm.groupSize))")
                }
            } else {
                print("  Bundle: unquantised (bf16) — LLM + DiT stay in plain Linear form")
            }
            // The "8-bit-full" variant emits a `dit_quantization` block to
            // override the DiT bits without affecting the LLM. The bf16 bundle
            // omits this; the loader will keep DiT as plain Linear.
            if let dit = json["dit_quantization"] as? [String: Any] {
                if let gs = dit["group_size"] as? Int { config.flow.dit.groupSize = gs }
                if let bits = dit["bits"] as? Int {
                    switch bits {
                    case 8:
                        config.flow.dit.bits = bits
                        print("  Bundle quantization (DiT): \(config.flow.dit.bits)-bit (group_size \(config.flow.dit.groupSize))")
                    case 16:
                        config.flow.dit.bits = bits
                        print("  Bundle precision (DiT): 16-bit (plain Linear; no .scales)")
                    default:
                        throw CosyVoiceTTSError.modelLoadFailed(
                            "CosyVoice DiT bundles must be 8-bit quantized or 16-bit/bf16 plain Linear.")
                    }
                } else {
                    print("  Bundle quantization (DiT): \(config.flow.dit.bits)-bit (group_size \(config.flow.dit.groupSize))")
                }
            }
        }
        let model = CosyVoiceTTSModel(config: config)

        // Load weights
        progressHandler?(0.5, "Loading LLM weights...")
        let llmURL = cacheDir.appendingPathComponent("llm.safetensors")
        try CosyVoiceWeightLoader.loadLLM(model.llm, from: llmURL)

        progressHandler?(0.7, "Loading flow weights...")
        let flowURL = cacheDir.appendingPathComponent("flow.safetensors")
        try CosyVoiceWeightLoader.loadFlow(model.flow, from: flowURL)

        progressHandler?(0.9, "Loading vocoder weights...")
        let hifiganURL = cacheDir.appendingPathComponent("hifigan.safetensors")
        try CosyVoiceWeightLoader.loadHiFiGAN(model.hifigan, from: hifiganURL)

        // Load tokenizer (Qwen2.5 BPE)
        progressHandler?(0.95, "Loading tokenizer...")
        let vocabURL = cacheDir.appendingPathComponent("vocab.json")
        try model.tokenizer.load(from: vocabURL)

        // Warmup: compile LLM and run dummy forward passes to pre-compile Metal shaders
        progressHandler?(0.98, "Warming up...")
        model.warmUp()

        MetalBudget.pinMemory()
        progressHandler?(1.0, "Model loaded")
        return model
    }

    /// Run minimal forward passes to compile Metal shaders and set up compiled generation.
    ///
    /// This eliminates first-inference latency from shader compilation (~200ms) and enables
    /// Metal kernel fusion for the LLM generation loop (~360 kernel dispatches fused)
    /// and DiT flow matching (~330 kernel dispatches × 10 ODE steps fused).
    public func warmUp() {
        // Shapeless compile fuses ~360 LLM kernel dispatches per step, but
        // MLX-Swift's tracer cannot infer the output shape of `addmm` under a
        // shapeless trace — that's the bias-fused matmul path that plain
        // `Linear` uses. Quantised bundles route attention/MLP through
        // `QuantizedLinear` (which uses `quantized_matmul + add` instead) so
        // they trace cleanly; the bf16 bundle's plain `Linear` does not.
        // When we detect a non-quantised LLM, skip compile entirely and run
        // the autoregressive loop through the direct `forwardStep` path
        // (still per-call eager, just no kernel fusion).
        let isLLMQuantized = (llm.layers.first?.selfAttn.qProj as? QuantizedLinear) != nil

        if isLLMQuantized {
            // Set up compiled LLM generation step (shapeless=true, traced on first call)
            llm.setupCompilation()
        }

        // Run a minimal prefill to compile all 24-layer attention + MLP shaders
        let textTokens: [Int32] = [2610]  // single token "You"
        let prefixEmbeds = llm.buildInputSequence(textTokens: textTokens)
        let (prefillLogits, warmupCache) = llm.forwardStep(
            prefixEmbeds, offset: MLXArray(Int32(0)), cache: nil)
        eval(prefillLogits)

        // Trace the compiled step with a single-token generation pass
        let warmupEmbed = llm.speechEmbedding(
            MLXArray([Int32(0)]).expandedDimensions(axis: 0))
        let (warmupLogits, _) = llm.executeStep(
            embeds: warmupEmbed, offset: prefixEmbeds.dim(1), cache: warmupCache)
        eval(warmupLogits)

        // The flow decoder uses a fixed-shape compile, so it traces cleanly
        // regardless of whether the DiT is quantised.
        flow.decoder.setupCompilation()
        flow.decoder.warmUp()
    }

    /// Synthesize speech from text (non-streaming).
    ///
    /// Returns: Array of float audio samples at 24kHz
    public func synthesize(
        text: String,
        language: String = "english",
        verbose: Bool = false
    ) -> [Float] {
        synthesize(text: text, language: language, instruction: "You are a helpful assistant.",
                   speakerEmbedding: nil, verbose: verbose)
    }

    /// Synthesize speech from text with a cloned voice.
    ///
    /// Uses a 192-dim CAM++ speaker embedding to condition the flow model,
    /// producing speech that mimics the voice characteristics of the embedding.
    public func synthesize(
        text: String,
        language: String = "english",
        speakerEmbedding: [Float],
        verbose: Bool = false
    ) -> [Float] {
        synthesize(text: text, language: language, instruction: "You are a helpful assistant.",
                   speakerEmbedding: speakerEmbedding, verbose: verbose)
    }

    /// Unified synthesis with both instruction and optional speaker embedding.
    ///
    /// - Parameters:
    ///   - text: Text to synthesize
    ///   - language: Target language
    ///   - instruction: Style instruction prefix (before `<|endofprompt|>`)
    ///   - speakerEmbedding: Optional 192-dim CAM++ speaker embedding
    ///   - verbose: Print timing info
    /// - Returns: Array of float audio samples at 24kHz
    public func synthesize(
        text: String,
        language: String = "english",
        instruction: String = "You are a helpful assistant.",
        speakerEmbedding: [Float]? = nil,
        promptToken: MLXArray? = nil,
        promptFeat: MLXArray? = nil,
        promptText: String? = nil,
        verbose: Bool = false
    ) -> [Float] {
        // 1. Tokenize text via Qwen2.5 BPE tokenizer. Track the content text
        //    length separately (without the instruction frame) so the LLM's
        //    min/max-len constraints scale to the actual content, not the
        //    instruction. Upstream: `min_len = (text_len - prompt_text_len) * ratio`.
        // Text frontend normalization (port of Python mlx_audio _normalize_text): zh punctuation,
        // blank handling, corner marks, bracket removal, trailing-comma→。; English digit spell-out.
        // Absent in the Swift port → pacing/punctuation gap vs the known-good Python path.
        let text = Self.normalizeText(text)
        let contentTokens = tokenizer.encode(text).map { Int32($0) }

        // For an unstyled zero-shot clone, upstream's text input is literally
        //   concat(transcript_tokens + [<|endofprompt|>], content_tokens)
        // with the reference speech tokens included in the LLM prefix. This
        // is the highest-fidelity identity path.
        //
        // A non-default instruction selects CosyVoice's `instruct2` layout:
        // the framed instruction becomes the LLM text prefix and the reference
        // speech tokens are withheld from the LLM. The Flow model still gets
        // promptToken + promptFeat, retaining the cloned-voice anchor while
        // allowing the LLM to follow the requested emotion or style.
        let useInstructionConditionedClone = Self.usesInstructionConditionedClone(
            promptText: promptText,
            instruction: instruction
        )
        let textTokens: [Int32]
        if useInstructionConditionedClone {
            textTokens = tokenizeText(text, language: language, instruction: instruction)
        } else if let pt = promptText, !pt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            // Match Python mlx_audio zero-shot layout EXACTLY:
            //   [ "You are a helpful assistant." ] [ <|endofprompt|> ] [ ref_transcript ] [ target ]
            // i.e. ZERO_SHOT_PROMPT_PREFIX = SYSTEM_PROMPT + <|endofprompt|>, then
            // concat(prompt_text = prefix+ref_transcript, target_text). The system frame MUST
            // precede <|endofprompt|>, and the ref transcript sits in the CONTENT region adjacent
            // to the target with NO separator. The appended reference speech (FSQ) tokens then act
            // as that transcript's acoustic prefix, so the model resumes at `target` and does NOT
            // re-vocalise the transcript.
            // The old layout `[ref_transcript] <|endofprompt|> [target]` (missing system frame,
            // <|endofprompt|> between transcript and target) pushed the transcript into the
            // instruction region → the acoustic prefix misaligned → the reference transcript got
            // spoken before the reply (short sentence 8.0 s vs 2.0 s). See tools cosy-diag.
            let systemTokens = tokenizer.encode(Self.assistantPrefix).map { Int32($0) }
            let promptTextTokens = tokenizer.encode(pt).map { Int32($0) }
            textTokens = systemTokens + [Self.endOfPromptToken] + promptTextTokens + contentTokens
        } else {
            textTokens = tokenizeText(text, language: language, instruction: instruction)
        }

        // 2. Generate speech tokens via LLM.
        //    For zero-shot cloning, the reference's FSQ codes are passed as
        //    `promptSpeechTokens` so the LLM's autoregressive state already
        //    encodes the target speaker before generation begins. Without this
        //    the LLM emits "neutral default voice" tokens that conflict with
        //    the flow's prompt_token + prompt_feat anchors and the cloned
        //    output drifts to a different voice (see PR #247).
        let promptSpeechTokensArr: [Int32]? = useInstructionConditionedClone
            ? nil
            : promptToken.map { pt in
                pt.reshaped(-1).asArray(Int32.self)
            }
        // Cap maxTokens proportionally to the content length. With prompt
        // conditioning the LLM is biased to "keep speaking" — a fixed cap of
        // 500 lets a 5-word phrase generate 20 s of repeats. Scale to 10×
        // content_tokens (with a sensible floor for very short content). This
        // mirrors upstream's `max_len = content_text_len * max_token_text_ratio`.
        let scaledMaxTokens = max(200, contentTokens.count * 10)
        var t0 = CFAbsoluteTimeGetCurrent()
        let speechTokens = llm.generate(
            textTokens: textTokens,
            promptSpeechTokens: promptSpeechTokensArr,
            contentTextLength: contentTokens.count,
            maxTokens: scaledMaxTokens
        )
        if verbose {
            let llmTime = CFAbsoluteTimeGetCurrent() - t0
            print(String(format: "  LLM: %.0fms (%d tokens, %.1fms/token)",
                         llmTime * 1000, speechTokens.count,
                         speechTokens.isEmpty ? 0 : llmTime * 1000 / Double(speechTokens.count)))
        }

        guard !speechTokens.isEmpty else {
            return []
        }

        // 3. Convert speech tokens to mel spectrogram via flow matching.
        //    When promptToken + promptFeat are supplied (the upstream zero-shot
        //    cloning path), the flow returns mel for the *full* prompt + generation
        //    span; we slice off the prompt region before HiFi-GAN.
        t0 = CFAbsoluteTimeGetCurrent()
        let tokenArray = MLXArray(speechTokens).expandedDimensions(axis: 0)  // [1, T]
        let spkEmb: MLXArray? = speakerEmbedding.map {
            MLXArray($0).expandedDimensions(axis: 0)
        }
        let fullMel = flow(
            tokens: tokenArray,
            spkEmbedding: spkEmb,
            promptToken: promptToken,
            promptFeat: promptFeat
        )
        eval(fullMel)

        // Pass the FULL mel (prompt + generation) to HiFi-GAN. Slicing the mel
        // here means HiFi-GAN's causal convolutions warm up against zero-padded
        // boundaries, producing a click/transient ~10 mel frames into the
        // generated audio (the convolutional receptive field). Instead we let
        // HiFi-GAN render both regions continuously and trim the prompt-region
        // audio AFTER, where the boundary is smooth.
        let mel = fullMel
        let promptAudioSamples: Int
        if let pf = promptFeat {
            let promptMelLen = pf.dim(2)
            // mel-rate is 50 Hz, audio sample rate is 24 kHz → 480 samples/mel
            promptAudioSamples = promptMelLen * (config.sampleRate / 50)
        } else {
            promptAudioSamples = 0
        }

        // Optional debug dump of the mel that reaches HiFi-GAN.
        if let dumpDir = ProcessInfo.processInfo.environment["COSY_DEBUG_DUMP_DIR"] {
            CosyVoiceDebugDump.tryWrite(mel, name: "swift_hifigan_input_mel", in: dumpDir)
        }

        if verbose {
            var path: [String] = []
            if speakerEmbedding != nil { path.append("spk") }
            if promptToken != nil { path.append("prompt_token") }
            if promptFeat != nil { path.append("prompt_feat") }
            let suffix = path.isEmpty ? "" : " (\(path.joined(separator: "+")))"
            print(String(format: "  Flow: %.0fms%@", (CFAbsoluteTimeGetCurrent() - t0) * 1000, suffix))
        }

        // 4. Convert mel to waveform via HiFi-GAN
        t0 = CFAbsoluteTimeGetCurrent()
        let audio = hifigan(mel)
        eval(audio)
        if verbose {
            print(String(format: "  HiFi-GAN: %.0fms", (CFAbsoluteTimeGetCurrent() - t0) * 1000))
        }

        // 5. Extract float samples, trimming the prompt-region audio so the
        //    caller only sees the synthesised content. The boundary inside
        //    HiFi-GAN's continuous render is smoother than a pre-sliced mel,
        //    so trimming here removes the slice-boundary click that appeared
        //    ~10 mel frames into the audio in the old path.
        var samples = audio.reshaped(-1).asArray(Float.self)
        if promptAudioSamples > 0 && promptAudioSamples < samples.count {
            samples = Array(samples[promptAudioSamples...])
        }
        return samples
    }

    /// Synthesize with streaming output.
    /// Synthesize speech asynchronously, yielding audio in chunks.
    ///
    /// Currently yields a single chunk with the full result (not yet
    /// incrementally streaming). The async stream interface is preserved
    /// for forward compatibility with chunked codec decoding.
    public func synthesizeStream(
        text: String,
        language: String = "english",
        chunkSize: Int = 25
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        let rate = config.sampleRate
        return AsyncThrowingStream { continuation in
            let task = Task { [weak self] in
                guard let self else {
                    continuation.finish()
                    return
                }
                do {
                    let samples = self.synthesize(text: text, language: language)
                    let chunk = AudioChunk(
                        samples: samples,
                        sampleRate: rate,
                        frameIndex: 0,
                        isFinal: true
                    )
                    continuation.yield(chunk)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// Token ID for `<|endofprompt|>` — added by CosyVoice3 but not in base tokenizer config.
    /// The text embedding table (151936 entries) includes this trained embedding at index 151646.
    static let endOfPromptToken: Int32 = 151646

    /// System-prompt frame that upstream uses in every official example.
    /// Custom style instructions are *appended* to this frame, not substituted
    /// for it — otherwise the model treats the instruction as content to speak.
    static let assistantPrefix = "You are a helpful assistant."

    /// Single source of truth for "is a style instruction present?".
    /// `instruction` defaults to `assistantPrefix`, so anything trimmed-equal to
    /// that (or empty) means "no style". This is the one place that knows the
    /// default sentinel — both the clone dispatch and the LLM framing defer to
    /// it instead of re-comparing the magic string.
    static func hasCustomStyleInstruction(_ instruction: String) -> Bool {
        let trimmed = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.isEmpty && trimmed != assistantPrefix
    }

    /// Frame an instruction for the LLM. CosyVoice3 was trained with the
    /// `"You are a helpful assistant. {style}"` system frame; a bare style is
    /// read aloud as content, so a custom style is prefixed with the frame while
    /// the default/empty case collapses to the frame alone.
    static func framedInstruction(_ instruction: String) -> String {
        guard hasCustomStyleInstruction(instruction) else { return assistantPrefix }
        let trimmed = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.hasPrefix(assistantPrefix) ? trimmed : "\(assistantPrefix) \(trimmed)"
    }

    /// Whether a cloned voice (one carrying a reference transcript) should use
    /// CosyVoice's `instruct2` layout instead of the transcript-led zero-shot
    /// path: only when a reference transcript is present *and* a custom style
    /// instruction was supplied. Otherwise the higher-fidelity transcript-led
    /// path is kept.
    static func usesInstructionConditionedClone(
        promptText: String?,
        instruction: String
    ) -> Bool {
        guard let promptText, !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return false
        }
        return hasCustomStyleInstruction(instruction)
    }

    /// Format and tokenize text for CosyVoice3 LLM: framed instruction, the
    /// `<|endofprompt|>` boundary, then the content tokens. Framing keeps the
    /// model in distribution — stripping the assistant prefix makes it read the
    /// style aloud instead of using it as conditioning.
    func tokenizeText(
        _ text: String, language: String,
        instruction: String = "You are a helpful assistant."
    ) -> [Int32] {
        let instructionTokens = tokenizer.encode(Self.framedInstruction(instruction)).map { Int32($0) }
        let textTokens = tokenizer.encode(text).map { Int32($0) }
        return instructionTokens + [Self.endOfPromptToken] + textTokens
    }

    // MARK: - Text normalization (port of Python mlx_audio cosyvoice3 `_normalize_text`)

    /// Lightweight frontend normalization matching the known-good Python path
    /// (cosyvoice3.py:1101-1126). No wetext/ttsfrd. Control-tag strings pass through.
    static func normalizeText(_ raw: String) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return trimmed }
        if trimmed.contains("<|") && trimmed.contains("|>") { return trimmed }  // control-tag bypass

        if containsChinese(trimmed) {
            var s = trimmed.replacingOccurrences(of: "\n", with: "")
            s = replaceBlank(s)
            s = s.replacingOccurrences(of: "²", with: "平方")
                 .replacingOccurrences(of: "³", with: "立方")
            s = s.replacingOccurrences(of: ".", with: "。")
            s = s.replacingOccurrences(of: " - ", with: "，")
            s = removeBracket(s)
            s = trailingSeparatorToPeriod(s)
            return s
        }
        return spellOutNumber(trimmed)
    }

    static func containsChinese(_ s: String) -> Bool {
        s.unicodeScalars.contains { (0x4E00...0x9FFF).contains($0.value) }
    }

    /// Drop spaces except a single space kept between two adjacent ASCII (non-space) chars.
    static func replaceBlank(_ text: String) -> String {
        let chars = Array(text)
        var out = ""
        for (i, ch) in chars.enumerated() {
            if ch != " " { out.append(ch); continue }
            if i == 0 || i == chars.count - 1 { continue }
            let prev = chars[i - 1], next = chars[i + 1]
            if prev.isASCII && prev != " " && next.isASCII && next != " " { out.append(ch) }
        }
        return out
    }

    static func removeBracket(_ text: String) -> String {
        var s = text
        for b in ["（", "）", "【", "】", "`"] { s = s.replacingOccurrences(of: b, with: "") }
        return s.replacingOccurrences(of: "——", with: " ")
    }

    /// Python: re.sub(r"[，,、]+$", "。", text) — a trailing run of those separators → single 。
    static func trailingSeparatorToPeriod(_ text: String) -> String {
        var s = text
        var stripped = false
        while let last = s.last, last == "，" || last == "," || last == "、" { s.removeLast(); stripped = true }
        if stripped { s.append("。") }
        return s
    }

    /// Spell out ASCII digit runs as English cardinal words (Python `_spell_out_number` via num2words).
    static func spellOutNumber(_ text: String) -> String {
        var out = "", digits = ""
        func flush() {
            if !digits.isEmpty {
                out += Int(digits).map { englishCardinal($0) } ?? digits
                digits = ""
            }
        }
        for ch in text {
            if ch.isASCII && ch.isNumber { digits.append(ch) } else { flush(); out.append(ch) }
        }
        flush()
        return out
    }

    static func englishCardinal(_ n: Int) -> String {
        if n == 0 { return "zero" }
        if n < 0 { return "minus " + englishCardinal(-n) }
        let ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                    "eighteen", "nineteen"]
        let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        func below1000(_ v: Int) -> String {
            var parts: [String] = []
            let h = v / 100, r = v % 100
            if h > 0 { parts.append(ones[h] + " hundred") }
            if r > 0 {
                if r < 20 { parts.append(ones[r]) }
                else {
                    let t = tens[r / 10]
                    parts.append(r % 10 == 0 ? t : t + "-" + ones[r % 10])
                }
            }
            return parts.joined(separator: " ")
        }
        var rem = n
        var parts: [String] = []
        for (val, name) in [(1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")] {
            if rem >= val { parts.append(below1000(rem / val) + " " + name); rem %= val }
        }
        if rem > 0 { parts.append(below1000(rem)) }
        return parts.joined(separator: " ")
    }
}
