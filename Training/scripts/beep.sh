#!/bin/bash
# Audible signal on the Windows side (WSL): plays tada.wav through the default
# audio device (console beeps do not reach headphones). Repeated N times (default 2).
n=${1:-2}
for i in $(seq 1 $n); do
  timeout 20 powershell.exe -NoProfile -Command "(New-Object Media.SoundPlayer 'C:\Windows\Media\tada.wav').PlaySync()" >/dev/null 2>&1
done
