#!/usr/bin/env bash
# Apply the reorg mapping in /tmp/reorg_mapping.csv.
# Usage:
#   bash tools/reorg_apply.sh dry      # show what would happen
#   bash tools/reorg_apply.sh apply    # actually move
#   bash tools/reorg_apply.sh undo     # restore using tools/reorg_undo.sh
set -euo pipefail

WS=/home/txy/lb/lean_workspace
CSV=/tmp/reorg_mapping.csv
UNDO=/home/txy/lb/tools/reorg_undo.sh

mode=${1:-dry}

if [[ "$mode" == "undo" ]]; then
    if [[ ! -f "$UNDO" ]]; then
        echo "no undo script at $UNDO"; exit 1
    fi
    bash "$UNDO"
    exit 0
fi

# Prep undo log
if [[ "$mode" == "apply" ]]; then
    {
        echo "#!/usr/bin/env bash"
        echo "# Auto-generated undo script. Reverses moves made by reorg_apply.sh."
        echo "set -euo pipefail"
    } > "$UNDO"
    chmod +x "$UNDO"
fi

# Create module dirs
modules=$(tail -n +2 "$CSV" | awk -F, '{print $2}' | sort -u)
for m in $modules; do
    if [[ "$m" == "_uncertain" ]]; then continue; fi
    if [[ "$mode" == "apply" ]]; then
        mkdir -p "$WS/$m"
    else
        echo "[dry] mkdir -p $WS/$m"
    fi
done

moved=0
skipped=0
while IFS=, read -r dirname module reason; do
    if [[ "$dirname" == "dirname" ]]; then continue; fi  # header
    if [[ "$module" == "_uncertain" ]]; then
        skipped=$((skipped+1)); continue
    fi
    src="$WS/$dirname"
    dst="$WS/$module/$dirname"
    if [[ ! -d "$src" ]]; then
        echo "SKIP missing: $src"; skipped=$((skipped+1)); continue
    fi
    if [[ -e "$dst" ]]; then
        echo "SKIP exists: $dst"; skipped=$((skipped+1)); continue
    fi
    if [[ "$mode" == "apply" ]]; then
        mv "$src" "$dst"
        echo "mv \"$dst\" \"$src\"" >> "$UNDO"
        moved=$((moved+1))
    else
        echo "[dry] mv $src -> $dst"
        moved=$((moved+1))
    fi
done < "$CSV"

echo ""
echo "mode=$mode  moved=$moved  skipped=$skipped"
if [[ "$mode" == "apply" ]]; then
    echo "Undo: bash $UNDO  (or  bash tools/reorg_apply.sh undo)"
fi
