#!/bin/sh

# start ollama serve & wait
ollama serve &
sleep 2

# local model list
ollama pull llama2
ollama pull llama3

# stay running
tail -f /dev/null