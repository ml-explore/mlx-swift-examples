#!/bin/sh

./mlx-run llm-tool list llms | \
	awk '{printf "./mlx-run llm-tool eval --download ~/Downloads/huggingface --model %s\n", $0}' | \
	awk '{printf "echo\necho ======\necho '\''%s'\''\n%s\n", $0, $0}'

./mlx-run llm-tool list vlms | \
	awk '{printf "./mlx-run llm-tool eval --download ~/Downloads/huggingface --model %s --resize 512 --image /tmp/test.jpg\n", $0}' | \
	awk '{printf "echo\necho ======\necho '\''%s'\''\n%s\n", $0, $0}'
