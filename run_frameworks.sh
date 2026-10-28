#!/usr/bin/env bash
# 只运行 LangChain / LangGraph / Deep Agents 专题；默认不调用模型。
# bash run_frameworks.sh --offline
# 编辑 .env 后运行：bash run_frameworks.sh --all
set -euo pipefail
cd "$(dirname "$0")"

mode="${1:---offline}"
if [[ $# -gt 1 || ( "$mode" != "--offline" && "$mode" != "--all" ) ]]; then
  echo "用法：bash run_frameworks.sh [--offline|--all]" >&2
  exit 2
fi

files=(
  p13b_langchain_chain.py
  p13c_langchain_structured.py
  p13d_langchain_agent.py
  p13a_langgraph_basics.py
  p13e_langgraph_react.py
  p13f_langgraph_map_reduce.py
  p13g_langgraph_persistence.py
  p13h_langgraph_store.py
  p13j_langgraph_time_travel.py
  p13i_deepagents_harness.py
)

for file in "${files[@]}"; do
  if [[ "$mode" == "--offline" ]]; then
    case "$file" in
      p13b_*|p13c_*|p13d_*|p13e_*|p13i_*) continue ;;
    esac
  fi
  printf '\n===== %s =====\n' "$file"
  uv run "$file" < /dev/null
done
echo "专题示例运行完成"
