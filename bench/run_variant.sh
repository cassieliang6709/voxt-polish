#!/bin/zsh
# Usage: run_variant.sh <label> <extra llama-server args...>
# Starts a temporary llama-server on 8099 with the production base args plus
# the extras, runs voxt_like.py, then stops it.
label=$1; shift
G=${MODEL:?Set MODEL=/path/to/model.gguf}
B=${0:A:h}
env ${=VARENV} llama-server -m $G -ngl 99 -np 2 -kvu --port 8099 --load-mode mmap+mlock "$@" > $B/server_$label.log 2>&1 &
P=$!
for i in $(seq 1 90); do curl -sf -m 1 localhost:8099/health >/dev/null && break; sleep 1; done
python3 -I $B/voxt_like.py http://localhost:8099/v1 $label ${ROUNDS:-2}
kill $P; wait $P 2>/dev/null; sleep 2
