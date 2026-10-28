#!/usr/bin/env bash
# 默认批量演示需要真实模型，会产生 API 费用。--check-b 只检查 B 批的本地控制规则。
# 可选集成：--with-retrieval 会下载 Embedding；--with-tracing 会上传公开教学追踪。
set -euo pipefail
cd "$(dirname "$0")"
check_b=false
retrieval=false
tracing=false
for option in "$@"; do
  case "$option" in
    --check-b) check_b=true ;;
    --with-retrieval) retrieval=true ;;
    --with-tracing) tracing=true ;;
    --help|-h)
      echo "用法：bash run_all.sh [--check-b | --with-retrieval --with-tracing]"
      echo "--check-b：只跑 B 批 7 个 --check，不调用模型、不下载 Embedding、不上传追踪。"
      exit 0 ;;
    *) echo "未知参数：$option" >&2; exit 2 ;;
  esac
done
if $check_b && { $retrieval || $tracing; }; then
  echo "--check-b 不能与集成开关一起使用" >&2
  exit 2
fi
for f in p*.py; do
  args=("$f")
  if $check_b; then
    case "$f" in
      p03a_*|p05b_*|p16b_*|p18a_*|p18b_*|p19_*|p26_*) args+=(--check) ;;
      *) continue ;;
    esac
  else
    case "$f" in
      p11a_*) continue ;;  # 由 MCP Client 启动
      p18a_*|p18b_*)
        if ! $retrieval; then echo "跳过 $f（单独运行或加 --with-retrieval）"; continue; fi ;;
      p26_*)
        if ! $tracing; then echo "跳过 $f（单独运行 --upload 或加 --with-tracing）"; continue; fi
        args+=(--upload) ;;
    esac
  fi
  printf '%-40s' "$f"
  start=$(date +%s)
  if uv run --quiet "${args[@]}" < /dev/null > "/tmp/run_all_${f%.py}.log" 2>&1; then
    echo "通过（$(( $(date +%s) - start )) 秒）"
  else
    echo "失败，输出见 /tmp/run_all_${f%.py}.log"
    exit 1
  fi
done
echo "本次选中的示例全部通过（不含跳过项）"
