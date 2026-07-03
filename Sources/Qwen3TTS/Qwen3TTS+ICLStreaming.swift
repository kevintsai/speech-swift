import Foundation
import MLX
import AudioCommon

// MARK: - Streaming ICL Voice Cloning
//
// Combines two capabilities the base fork keeps separate:
//   • `synthesizeStream`            → real streaming (low TTFA), but preset speakers only.
//   • `synthesizeWithVoiceCloneICL` → clone an arbitrary reference voice, but blocking only.
//
// This adds the missing combination: stream a cloned voice.
//
// The key idea is the decode boundary. The blocking ICL path prepends the reference codec
// for the decode, then cuts the reference fraction back off by a proportional sample estimate
// (a lossy slice at a non-zero-crossing). Streaming avoids that entirely: the reference codec
// becomes the causal Mimi decoder's LEFT-CONTEXT for the first target chunk(s) — decoded for
// warmup context but never emitted (we keep only the target chunk's samples from the right).
// So there is no cold-start artifact and no proportional cut; the reference is pure context.
extension Qwen3TTSModel {

    /// Stream a target text spoken in a cloned reference voice (ICL), yielding audio chunks
    /// as they are generated. Same voice-fidelity path as `synthesizeWithVoiceCloneICL`, but
    /// with `synthesizeStream`-style low first-packet latency.
    ///
    /// - Parameters:
    ///   - text: Target text to synthesize.
    ///   - referenceAudio: Reference speaker PCM (any sample rate; resampled to 24kHz internally).
    ///   - referenceSampleRate: Sample rate of `referenceAudio`.
    ///   - referenceText: Exact transcript of the reference recording.
    ///   - language: Language hint; "auto" (default) matches the QwenLM/mlx-audio reference.
    ///   - sampling: Sampling config (repetition penalty auto-bumped under sampling, as in blocking ICL).
    ///   - codecEncoder: `SpeechTokenizerEncoder` from `fromPretrainedWithEncoder()`.
    ///   - streaming: Chunk sizes / decoder left-context.
    /// - Returns: An async stream of `AudioChunk` (target audio only; reference never emitted).
    public func synthesizeStreamWithVoiceCloneICL(
        text: String,
        referenceAudio: [Float],
        referenceSampleRate: Int = 24000,
        referenceText: String,
        language: String = "auto",
        sampling: SamplingConfig = .default,
        codecEncoder: SpeechTokenizerEncoder,
        streaming: StreamingConfig = .default
    ) -> AsyncThrowingStream<AudioChunk, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try self.runStreamingICLGeneration(
                        text: text, referenceAudio: referenceAudio, referenceSampleRate: referenceSampleRate,
                        referenceText: referenceText, language: language, sampling: sampling,
                        codecEncoder: codecEncoder, streaming: streaming, continuation: continuation)
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { @Sendable _ in task.cancel() }
        }
    }

    /// Internal streaming loop. Mirrors `runStreamingGeneration` but with an ICL prefill
    /// (reference codec in the talker context) and an ICL chunk decode (reference codec as
    /// decoder left-context, never emitted).
    func runStreamingICLGeneration(
        text: String,
        referenceAudio: [Float],
        referenceSampleRate: Int,
        referenceText: String,
        language: String,
        sampling: SamplingConfig,
        codecEncoder: SpeechTokenizerEncoder,
        streaming: StreamingConfig,
        continuation: AsyncThrowingStream<AudioChunk, Error>.Continuation
    ) throws {
        guard let tokenizer = tokenizer else { throw TTSError.tokenizerNotLoaded }

        // "auto" → skip language-id token (codec_nothink branch), matching blocking ICL.
        let langId: Int?
        let normalized = language.lowercased()
        if normalized == "auto" || normalized.isEmpty {
            langId = nil
        } else if let id = CodecTokens.languageId(for: language) {
            langId = id
        } else {
            langId = nil   // unknown → fall back to auto rather than fail mid-stream
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        let samplesPerFrame = 1920  // 24000 / 12.5

        // Reference codec (cached per reference).
        let refCodes: MLXArray
        if let cached = referenceAudioCache.codecRefCodes(for: referenceAudio, sampleRate: referenceSampleRate) {
            refCodes = cached
        } else {
            let audio24k = referenceSampleRate == 24000
                ? referenceAudio
                : AudioFileLoader.resample(referenceAudio, from: referenceSampleRate, to: 24000)
            let codes = codecEncoder.encode(samples: audio24k)
            eval(codes)
            referenceAudioCache.storeCodecRefCodes(codes, audio: referenceAudio, sampleRate: referenceSampleRate)
            refCodes = codes
        }

        // Speaker embedding (cached) — ICL still uses x-vector conditioning, same as blocking.
        let speakerEmbed: MLXArray
        if let cached = referenceAudioCache.speakerEmbed(for: referenceAudio, sampleRate: referenceSampleRate) {
            speakerEmbed = cached
        } else {
            let mels = SpeakerMel.compute(audio: referenceAudio, sampleRate: referenceSampleRate)
            let embed = speakerEncoder(mels)
            eval(embed)
            referenceAudioCache.storeSpeakerEmbed(embed, audio: referenceAudio, sampleRate: referenceSampleRate)
            speakerEmbed = embed
        }

        // ICL prefill (reference codec context prepended into the AR sequence).
        let (prefillEmbeds, trailingTextHidden, ttsPadEmbed) = buildICLPrefillEmbeddings(
            refCodes: refCodes, referenceText: referenceText, targetText: text,
            language: language, languageId: langId, speakerEmbed: speakerEmbed, tokenizer: tokenizer)
        eval(prefillEmbeds, trailingTextHidden, ttsPadEmbed)

        // Sampling: bump repetition penalty under sampling + cap maxTokens (prevents the
        // under-EOS runaway / dragging seen on some lines), matching blocking ICL.
        var iclSampling = sampling
        if iclSampling.temperature > 0 && iclSampling.repetitionPenalty < 1.5 {
            iclSampling.repetitionPenalty = 1.5
        }
        let targetTokenCount = tokenizer.encode(text).count
        let textDerivedCap = max(96, targetTokenCount * 8)
        iclSampling.maxTokens = min(iclSampling.maxTokens, textDerivedCap)
        let safeMaxTokens = iclSampling.maxTokens

        // Reference codec as [[Int32]] (per code group) for decoder left-context.
        let numGroups = config.codePredictor.numCodeGroups
        let tRef = refCodes.dim(2)
        let refFlat = refCodes.asType(.int32).asArray(Int32.self)   // row-major [1, numGroups, tRef]
        var refCodebooks: [[Int32]] = []
        refCodebooks.reserveCapacity(numGroups)
        for g in 0..<numGroups {
            var col = [Int32](); col.reserveCapacity(tRef)
            let base = g * tRef
            for t in 0..<tRef { col.append(refFlat[base + t]) }
            refCodebooks.append(col)
        }

        let cpSamplingConfig = SamplingConfig(temperature: iclSampling.temperature, topK: iclSampling.topK)
        let prefillLen = prefillEmbeds.dim(1)

        // Prefill.
        var (logits, hiddenStates, newCache) = talker(
            inputsEmbeds: prefillEmbeds, offset: MLXArray(Int32(0)), cache: nil)
        var talkerCache = newCache

        var nextToken = sampleToken(
            logits: logits[0..., (prefillLen - 1)..<prefillLen, 0...],
            config: iclSampling, generatedTokens: [],
            suppressRange: (2048, 3072), eosTokenId: CodecTokens.codecEos)

        if nextToken == Int32(CodecTokens.codecEos) {
            continuation.yield(AudioChunk(samples: [], sampleRate: 24000, frameIndex: 0,
                                          isFinal: true, elapsedTime: CFAbsoluteTimeGetCurrent() - t0))
            return
        }

        var generatedFirstCodebook: [Int32] = [nextToken]
        var generatedAllCodebooks: [[Int32]] = (0..<numGroups).map { _ in [] }
        generatedAllCodebooks[0].append(nextToken)

        let lastHidden = hiddenStates[0..., (prefillLen - 1)..<prefillLen, 0...]
        var codeTokens = predictCodebooksForTimestep(
            hiddenState: lastHidden, firstCodebookToken: nextToken, cpSamplingConfig: cpSamplingConfig)
        for (i, token) in codeTokens.enumerated() { generatedAllCodebooks[i + 1].append(token) }

        var trailingIdx = 0
        var step = prefillLen
        var emittedFrames = 0
        var emittedFinal = false
        var nextEmitThreshold = streaming.firstChunkFrames

        // Decode [start, end) target frames with reference codec as left-context; yield only target audio.
        // The FINAL chunk gets right-context tail-pad so the causal vocoder renders the last syllable's
        // tail cleanly (otherwise the last word/字 gets pitch wobble / noise / cut — worst on English).
        func emit(_ start: Int, _ end: Int, isFinal: Bool) {
            let chunk = decodeICLStreamChunk(
                refCodebooks: refCodebooks, targetCodebooks: generatedAllCodebooks,
                chunkStart: start, chunkEnd: end,
                decoderLeftContext: streaming.decoderLeftContext, samplesPerFrame: samplesPerFrame,
                tailPadFrames: isFinal ? 3 : 0)
            continuation.yield(AudioChunk(samples: chunk, sampleRate: 24000, frameIndex: start,
                                          isFinal: isFinal, elapsedTime: CFAbsoluteTimeGetCurrent() - t0))
        }

        if generatedFirstCodebook.count >= nextEmitThreshold {
            emit(0, generatedFirstCodebook.count, isFinal: false)
            emittedFrames = generatedFirstCodebook.count
            nextEmitThreshold = emittedFrames + streaming.chunkFrames
        }

        for iterIdx in 1..<safeMaxTokens {
            if Task.isCancelled { return }

            let textEmbed: MLXArray
            let trailingLen = trailingTextHidden.dim(1)
            if trailingIdx < trailingLen {
                textEmbed = trailingTextHidden[0..., trailingIdx..<(trailingIdx + 1), 0...]
                trailingIdx += 1
            } else {
                textEmbed = ttsPadEmbed
            }

            let codecEmbed = talker.embedCodec(MLXArray([nextToken]).expandedDimensions(axis: 0))
                + codePredictor.batchEmbedAllGroups(codeTokens)
            let stepEmbeds = textEmbed + codecEmbed

            (logits, hiddenStates, newCache) = executeTalkerStep(embeds: stepEmbeds, offset: step, cache: talkerCache)
            talkerCache = newCache

            nextToken = sampleToken(
                logits: logits, config: iclSampling, generatedTokens: generatedFirstCodebook,
                suppressRange: (2048, 3072), eosTokenId: CodecTokens.codecEos)
            let isEos = nextToken == Int32(CodecTokens.codecEos)

            if !isEos {
                generatedFirstCodebook.append(nextToken)
                generatedAllCodebooks[0].append(nextToken)
                codeTokens = predictCodebooksForTimestep(
                    hiddenState: hiddenStates, firstCodebookToken: nextToken, cpSamplingConfig: cpSamplingConfig)
                for (i, token) in codeTokens.enumerated() { generatedAllCodebooks[i + 1].append(token) }
            }

            step += 1
            let totalFrames = generatedFirstCodebook.count
            let shouldEmit = isEos || totalFrames >= nextEmitThreshold || iterIdx == safeMaxTokens - 1
            if shouldEmit && totalFrames > emittedFrames {
                let isFinalChunk = isEos || iterIdx == safeMaxTokens - 1
                emit(emittedFrames, totalFrames, isFinal: isFinalChunk)
                if isFinalChunk { emittedFinal = true }
                emittedFrames = totalFrames
                nextEmitThreshold = emittedFrames + streaming.chunkFrames
            }

            if isEos { break }
        }

        let numFrames = generatedFirstCodebook.count
        if numFrames >= safeMaxTokens && nextToken != Int32(CodecTokens.codecEos) {
            let estSec = Double(numFrames) / 12.5
            print("Warning: ICL stream hit safety limit of \(safeMaxTokens) tokens (~\(String(format: "%.1f", estSec))s audio).")
        }

        if emittedFrames < numFrames {
            emit(emittedFrames, numFrames, isFinal: true)
            emittedFinal = true
        }
        if !emittedFinal {
            continuation.yield(AudioChunk(samples: [], sampleRate: 24000, frameIndex: emittedFrames,
                                          isFinal: true, elapsedTime: CFAbsoluteTimeGetCurrent() - t0))
        }
    }

    /// Decode target frames `[chunkStart, chunkEnd)` using the reference codec as decoder
    /// left-context (backfilled from the reference tail when prior target frames are insufficient,
    /// e.g. the first chunk). Only the target chunk's samples are returned; the reference /
    /// context samples are trimmed from the right, so the reference is never emitted.
    func decodeICLStreamChunk(
        refCodebooks: [[Int32]],
        targetCodebooks: [[Int32]],
        chunkStart: Int,
        chunkEnd: Int,
        decoderLeftContext: Int,
        samplesPerFrame: Int,
        tailPadFrames: Int = 0
    ) -> [Float] {
        let numGroups = targetCodebooks.count
        let refCount = refCodebooks.first?.count ?? 0
        let realChunkFrames = chunkEnd - chunkStart

        // Prefer prior target frames as context; backfill the rest from the reference tail.
        let tgtCtxStart = max(chunkStart - decoderLeftContext, 0)
        let tgtCtxFrames = chunkStart - tgtCtxStart
        let refCtxFrames = min(max(decoderLeftContext - tgtCtxFrames, 0), refCount)

        var codebookArrays: [MLXArray] = []
        codebookArrays.reserveCapacity(numGroups)
        for g in 0..<numGroups {
            var slice: [Int32] = []
            slice.reserveCapacity(refCtxFrames + (chunkEnd - tgtCtxStart) + tailPadFrames)
            if refCtxFrames > 0 {
                slice.append(contentsOf: refCodebooks[g][(refCount - refCtxFrames)..<refCount])
            }
            slice.append(contentsOf: targetCodebooks[g][tgtCtxStart..<chunkEnd])
            // Right-context: repeat the last real frame so the causal Mimi decoder has look-ahead
            // and renders the final syllable's tail cleanly. These padded samples are dropped below.
            if tailPadFrames > 0, chunkEnd > 0 {
                let last = targetCodebooks[g][chunkEnd - 1]
                slice.append(contentsOf: Array(repeating: last, count: tailPadFrames))
            }
            codebookArrays.append(MLXArray(slice).expandedDimensions(axis: 0))  // [1, T]
        }
        var codes = stacked(codebookArrays, axis: 1)  // [1, numGroups, T]

        // Zero-pad if fewer than 4 frames (ConvNeXt kernel=7 minimum). With reference context
        // this is only hit when the reference itself is tiny.
        let minDecodeFrames = 4
        let realFrames = codes.dim(2)
        if realFrames < minDecodeFrames {
            let pad = MLXArray.zeros([1, numGroups, minDecodeFrames - realFrames]).asType(.int32)
            codes = concatenated([pad, codes], axis: 2)
        }

        let waveform = codecDecoder.executeDecoder(codes)  // [1, T_samples, 1]

        // Keep only the target chunk. Drop the tail-pad's samples from the right (it existed only to
        // give the last real frame right-context), then keep the last realChunkFrames*samplesPerFrame
        // samples of what remains (keeping from the right absorbs the decoder's left-side warmup).
        let expectedKept = realChunkFrames * samplesPerFrame
        let totalSamples = waveform.dim(1)
        let end = max(0, totalSamples - tailPadFrames * samplesPerFrame)
        let start = max(0, end - expectedKept)
        let kept = waveform[0..., start..<end, 0...]
        let flat = kept.squeezed()
        eval(flat)
        return flat.asArray(Float.self)
    }
}
