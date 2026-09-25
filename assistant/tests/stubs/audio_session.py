class PhraseChunker:
    def __init__(self): self.buf = ""
    def feed(self, text, final=False):
        self.buf += text
        out = []
        import re
        while True:
            m = re.search(r"[.!?]\s", self.buf)
            if not m: break
            out.append(self.buf[:m.end()].strip()); self.buf = self.buf[m.end():]
        if final and self.buf.strip():
            out.append(self.buf.strip()); self.buf = ""
        return out
class StreamingWavePcmDecoder:
    def feed(self, chunk, final=False): yield chunk
class AudioSession: pass
