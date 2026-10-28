#!/usr/bin/env bash
# 把所有示例依次跑一遍（需配置真实模型，会产生 API 费用）。任何一个出错就停下并报出文件名。
# 用法：bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
for f in p*.py; do
  case "$f" in
    p11a_*) continue ;;  # MCP Server 本身不单独跑，由 p11b 启动
  esac
  printf '%-40s' "$f"
  start=$(date +%s)
  if uv run --quiet "$f" < /dev/null > "/tmp/run_all_${f%.py}.log" 2>&1; then
    echo "通过（$(( $(date +%s) - start )) 秒）"
  else
    echo "失败，输出见 /tmp/run_all_${f%.py}.log"
    exit 1
  fi
done
echo "全部通过"
