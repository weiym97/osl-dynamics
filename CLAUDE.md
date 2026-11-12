# osl-dynamics

## Memory Mirror

Whenever you create, edit or delete a memory file (including `MEMORY.md`), mirror every project's memory folder to `/well/win-fmrib-analysis/users/uap971/claude_context/memory/` in the same turn:

```bash
for d in "$HOME"/.claude/projects/*/memory; do
  p="${d%/memory}"; p="${p##*/}"; p="${p#-gpfs3-well-win-fmrib-analysis-users-uap971-}"
  rsync -a --delete "$d/" "/well/win-fmrib-analysis/users/uap971/claude_context/memory/$p/"
done
```
