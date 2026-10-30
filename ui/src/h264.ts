// H.264 video from the server's /video/h264/... streams, decoded with WebCodecs onto a canvas.
// The graphics chip decodes it on most devices, far cheaper than decoding MJPEG frames.
// Stream format (sim/flooduav/bridge.py /h264): [4-byte big-endian length][1 byte: 1 key frame][Annex B access unit].

/** WebCodecs is there: Chrome, Edge, Safari 16.4+, Firefox 130+. Otherwise the panels show MJPEG. */
export const h264Supported = typeof window !== "undefined" && "VideoDecoder" in window;

/** The codec string from the SPS of a key frame, e.g. "avc1.4d401f", or null without one. */
function codecOf(au: Uint8Array): string | null {
  for (let i = 0; i + 6 < au.length; i++) {
    if (au[i] === 0 && au[i + 1] === 0 && au[i + 2] === 1 && (au[i + 3] & 0x1f) === 7) {
      const hex = (b: number) => b.toString(16).padStart(2, "0");
      return `avc1.${hex(au[i + 4])}${hex(au[i + 5])}${hex(au[i + 6])}`;
    }
  }
  return null;
}

/**
 * Plays `url` on `canvas` until the returned stop function is called.
 * `onEnd(fatal)`: fatal means H.264 cannot play here (no such stream, codec unsupported), so use MJPEG;
 * otherwise the stream broke or ended and can be tried again.
 */
export function playH264(url: string, canvas: HTMLCanvasElement, onEnd: (fatal: boolean) => void): () => void {
  const ctl = new AbortController();
  const ctx = canvas.getContext("2d");
  let dec: VideoDecoder | null = null;
  let ended = false;
  const end = (fatal: boolean) => {
    if (!ended && !ctl.signal.aborted) {
      ended = true;
      onEnd(fatal);
    }
  };

  (async () => {
    const r = await fetch(url, { signal: ctl.signal, cache: "no-store" });
    if (!r.ok || r.headers.get("content-type") !== "video/h264" || !r.body) return end(true);
    const reader = r.body.getReader();
    let buf = new Uint8Array(0);
    let n = 0;
    let waitKey = true;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return end(false);
      const next = new Uint8Array(buf.length + value.length);
      next.set(buf);
      next.set(value, buf.length);
      buf = next;
      let at = 0;
      while (buf.length - at >= 5) {
        const len = new DataView(buf.buffer, buf.byteOffset + at, 4).getUint32(0);
        if (buf.length - at - 5 < len) break;
        const key = buf[at + 4] === 1;
        const au = buf.subarray(at + 5, at + 5 + len);
        at += 5 + len;
        if (!dec) {
          if (!key) continue;
          const codec = codecOf(au);
          const config = { codec: codec ?? "", optimizeForLatency: true };
          if (!codec || !(await VideoDecoder.isConfigSupported(config)).supported) return end(true);
          dec = new VideoDecoder({
            output: (f) => {
              if (canvas.width !== f.displayWidth || canvas.height !== f.displayHeight) {
                canvas.width = f.displayWidth;
                canvas.height = f.displayHeight;
              }
              ctx?.drawImage(f, 0, 0);
              f.close();
            },
            error: () => end(false),
          });
          dec.configure(config);
        }
        if (dec.state !== "configured") return end(false);
        if (dec.decodeQueueSize > 4) waitKey = true; // the decoder falls behind: skip to the next key frame
        if (waitKey && !key) continue;
        waitKey = false;
        dec.decode(new EncodedVideoChunk({ type: key ? "key" : "delta", timestamp: n++ * 100_000, data: au }));
      }
      buf = buf.slice(at);
    }
  })().catch(() => end(false));

  return () => {
    ctl.abort();
    if (dec && dec.state !== "closed") dec.close();
  };
}
