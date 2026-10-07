#!/bin/bash
# Download PMXT archive with auto-resume. Run in a separate terminal:
#   bash scripts/download_pmxt.sh
#
# ~36GB download from s3.jbecker.dev. Server is flaky — this script
# keeps retrying until complete.

cd "$(dirname "$0")/../data/pmxt" || exit 1
EXPECTED_SIZE=36020641508

echo "Downloading PMXT archive (36GB) to data/pmxt/data.tar.zst"
echo "Will auto-resume on connection drops."
echo ""

while true; do
  CURRENT=$(stat -f%z data.tar.zst 2>/dev/null || echo 0)
  if [ "$CURRENT" -ge "$EXPECTED_SIZE" ]; then
    echo ""
    echo "Download complete! ($CURRENT bytes)"
    echo ""
    echo "Next steps:"
    echo "  cd data/pmxt"
    echo "  zstd -d data.tar.zst -o data.tar"
    echo "  tar xf data.tar"
    echo "  cd ../.. && .venv/bin/python scripts/analyze_pmxt.py"
    exit 0
  fi

  PCT=$(python3 -c "print(f'{$CURRENT/$EXPECTED_SIZE*100:.0f}')" 2>/dev/null || echo "?")
  echo "Progress: $(python3 -c "print(f'{$CURRENT/1e9:.1f}')")GB / 36.0GB (${PCT}%) — resuming..."

  curl -L -C - --retry 3 --retry-delay 10 --connect-timeout 30 \
    --max-time 300 -o data.tar.zst \
    "https://s3.jbecker.dev/data.tar.zst" 2>/dev/null

  sleep 5
done
