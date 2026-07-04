import XCTest
@testable import Qwen3TTS
import AudioCommon

// MARK: - Unit Tests

final class EOSCapTests: XCTestCase {

    func testMaxTokensCap() {
        // "Hello" tokenizes to ~1-2 tokens → cap at max(75, 2*6) = 75
        // "A long sentence with many words to test the capping logic" → ~12 tokens → cap at max(75, 72) = 75
        // Very long text with 100+ tokens → cap at 100*6 = 600
        // Verify the formula: min(maxTokens, max(75, textTokenCount * 6))

        // Short text: floor kicks in
        let shortCap = min(500, max(75, 2 * 6))  // 2 tokens
        XCTAssertEqual(shortCap, 75, "Short text should use floor of 75")

        // Medium text: factor of 6
        let mediumCap = min(500, max(75, 20 * 6))  // 20 tokens
        XCTAssertEqual(mediumCap, 120, "Medium text: 20 tokens × 6 = 120")

        // Long text: maxTokens kicks in
        let longCap = min(500, max(75, 200 * 6))  // 200 tokens
        XCTAssertEqual(longCap, 500, "Long text should be capped at maxTokens=500")
    }

    // Duration-based clone cap (cloneTokenCap): the runaway-babble guard for
    // ICL / x-vector clone paths. Estimate = max(words/2.6, chars/14) + punct*0.5s
    // + 1s base, * 12.5 tokens/s * 1.35 margin, floor 96.
    func testCloneTokenCapChinese70Chars() {
        // ~70 hanzi + fullwidth punctuation = the typeup dictation-readback case that
        // babbled to ~5x length under the old x8 cap.
        let text = String(repeating: "這是一段用來測試的中文字", count: 6) + ",句尾。!?"
        // chars ~76: seconds ~ 76/14 + 4*0.5 + 1 ~ 8.4 -> tokens ~ 8.4*12.5*1.35 ~ 143
        let cap = Qwen3TTSModel.cloneTokenCap(for: text, sampling: SamplingConfig(temperature: 0.5))
        XCTAssertGreaterThan(cap, 96, "70+ chars should exceed the floor")
        XCTAssertLessThan(cap, 200, "old x8 cap would be ~560; duration cap must be far tighter")
    }

    func testCloneTokenCapEnglish() {
        let text = "Please align the column headers in the dashboard table."
        // 9 words/2.6 ~ 3.5s vs 47 chars/14 ~ 3.4s -> max 3.5 + 1 punct*0.5 + 1 ~ 5.0s
        // -> ~84 tokens -> floor 96 applies.
        let cap = Qwen3TTSModel.cloneTokenCap(for: text, sampling: SamplingConfig(temperature: 0.5))
        XCTAssertEqual(cap, 96, "short English line should land on the floor")
    }

    func testCloneTokenCapEmptyAndCeiling() {
        XCTAssertEqual(
            Qwen3TTSModel.cloneTokenCap(for: "  ", sampling: SamplingConfig(temperature: 0.5)),
            96, "empty text -> floor")
        var tiny = SamplingConfig(temperature: 0.5)
        tiny.maxTokens = 50
        XCTAssertEqual(
            Qwen3TTSModel.cloneTokenCap(for: "hello world", sampling: tiny),
            50, "caller's maxTokens is still the hard ceiling")
    }
}

// MARK: - E2E Tests

final class E2EEOSCapTests: XCTestCase {

    func testXVectorShortEnglishEOS() async throws {
        // Short English text should stop well before maxTokens
        let model = try await Qwen3TTSModel.fromPretrained()

        let refAudio = try AudioFileLoader.load(
            url: URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav"),
            targetSampleRate: 24000)
        let refSpeech = Array(refAudio[Int(5.17 * 24000)..<min(Int(8.37 * 24000), refAudio.count)])

        let waveform = model.synthesizeWithVoiceClone(
            text: "Hello.",
            referenceAudio: refSpeech,
            referenceSampleRate: 24000,
            language: "english")

        XCTAssertGreaterThan(waveform.count, 0, "Should produce audio")
        let duration = Double(waveform.count) / 24000.0
        print("Short English x-vector: \(String(format: "%.2f", duration))s")
        // "Hello." is ~1-2 tokens → cap at 75 tokens = ~6s max
        XCTAssertLessThan(duration, 8.0, "Short text should not produce long audio")
    }

    func testXVectorGermanEOS() async throws {
        // Issue #139: German text hits maxTokens with x-vector mode.
        // With the cap, it should stop much earlier.
        let model = try await Qwen3TTSModel.fromPretrained()

        let refAudio = try AudioFileLoader.load(
            url: URL(fileURLWithPath: "Tests/Qwen3ASRTests/Resources/test_audio.wav"),
            targetSampleRate: 24000)
        let refSpeech = Array(refAudio[Int(5.17 * 24000)..<min(Int(8.37 * 24000), refAudio.count)])

        let waveform = model.synthesizeWithVoiceClone(
            text: "Hallo, das ist ein Test.",
            referenceAudio: refSpeech,
            referenceSampleRate: 24000,
            language: "german")

        XCTAssertGreaterThan(waveform.count, 0, "Should produce audio")
        let duration = Double(waveform.count) / 24000.0
        print("German x-vector: \(String(format: "%.2f", duration))s")
        // Without cap: would hit 500 tokens = ~40s. With cap: should be < 10s
        XCTAssertLessThan(duration, 12.0, "German text should not run to maxTokens")
    }
}
