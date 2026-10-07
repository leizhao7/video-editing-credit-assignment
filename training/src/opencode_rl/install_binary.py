"""Image-build-only installer for the reviewed official OpenCode release."""
from __future__ import annotations
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request

URL = "https://github.com/anomalyco/opencode/releases/download/v1.15.5/opencode-linux-x64-baseline.tar.gz"
ARCHIVE_SHA256 = "9d6a64ade5379f3642d4f5c519fa9bb1aab7e889a6a6c6ce8133da8b2c9763b3"


def main():
    with urllib.request.urlopen(URL, timeout=120) as response:
        body = response.read(60_000_000)
    if hashlib.sha256(body).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("Official release archive checksum mismatch")
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as archive:
        candidates = [m for m in archive.getmembers() if m.isfile() and Path(m.name).name == "opencode"]
        if len(candidates) != 1:
            raise RuntimeError("Unexpected release archive structure")
        binary = archive.extractfile(candidates[0]).read()
    target = Path("/usr/local/bin/opencode")
    with target.open("xb") as handle:
        handle.write(binary)
    target.chmod(0o755)
    print(json.dumps({"version": "1.15.5", "archive_sha256": ARCHIVE_SHA256,
                      "binary_sha256": hashlib.sha256(binary).hexdigest()}))


if __name__ == "__main__":
    main()
