/**
 * mic-processor.js — AudioWorkletProcessor
 * Runs in the AudioWorkletGlobalScope (dedicated audio thread).
 * Converts float32 PCM → int16 and posts the buffer to the main thread.
 */
class MicProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (!channel || channel.length === 0) return true;

    const s16 = new Int16Array(channel.length);
    for (let i = 0; i < channel.length; i++) {
      s16[i] = Math.max(-32768, Math.min(32767, channel[i] * 32768));
    }
    // Transfer the underlying buffer (zero-copy)
    this.port.postMessage(s16.buffer, [s16.buffer]);
    return true; // keep processor alive
  }
}

registerProcessor("mic-processor", MicProcessor);
