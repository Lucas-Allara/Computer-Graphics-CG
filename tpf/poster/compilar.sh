#!/bin/sh
# Compila poster.pdf (dos pasadas: los nodos con remember picture lo necesitan).
cd "$(dirname "$0")"
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
lualatex -interaction=nonstopmode -file-line-error poster.tex >/dev/null
lualatex -interaction=nonstopmode -file-line-error poster.tex >/dev/null
grep -E '^\./poster.tex:[0-9]+:|Overfull|undefined' poster.log
rm -f poster.aux poster.log poster.out
