#!/usr/bin/env bash
# zip_spatialdata.sh — pack each SpatialData Zarr into *.zarr.zip
# ------------------------------------------------------------------
set -euo pipefail

SRC_ROOT="/Volumes/DD1_FGS/MICS/data_HE2CellType/CT_DS/sdata_final"
DST_ROOT="/Volumes/DD1_FGS/MICS/data_HE2CellType/STHELAR/sdata_slides"

# ---- replace old symlink ----------------------------------------------------
if [[ -L "$DST_ROOT" ]]; then
    echo "Removing previous symlink $DST_ROOT"
    rm "$DST_ROOT"
fi
mkdir -p "$DST_ROOT"

# ---- util functions ---------------------------------------------------------
sizeof ()   { du -sk "$1" | awk '{printf "%.1f GiB",$1/1048576}'; }

secs_to_hms () {                       # $1 = seconds → HH:MM:SS
    printf "%02d:%02d:%02d" \
           $(( $1/3600 )) $(( ($1%3600)/60 )) $(( $1%60 ))
}

spinner () {                           # $1 = PID  $2 = log-file  $3 = total_files  $4 = t₀
    local pid=$1 log=$2 total=$3 start=$4
    local spin='-\|/' i=0 done_files pct eta elapsed

    while kill -0 "$pid" 2>/dev/null; do
        now=$(date +%s)
        elapsed=$(( now - start ))

        # how many “adding:” lines so far?
        done_files=$(awk '/^  adding:/{c++} END{print c+0}' "$log" 2>/dev/null)

        if (( done_files > 0 )); then
            pct=$(( 100 * done_files / total ))
            eta=$(( elapsed * (total - done_files) / done_files ))
            printf "\r[%c] %3d%%  %s elapsed, ~%s left" \
                   "${spin:i++%4:1}" "$pct" \
                   "$(secs_to_hms "$elapsed")" "$(secs_to_hms "$eta")"
        else
            printf "\r[%c] %s elapsed…" \
                   "${spin:i++%4:1}" "$(secs_to_hms "$elapsed")"
        fi
        sleep 1
    done
    rm -f "$log"
}

# ---- main loop --------------------------------------------------------------
shopt -s nullglob
for zdir in "$SRC_ROOT"/*.zarr; do
    slide=$(basename "$zdir" .zarr)
    zipout="$DST_ROOT/${slide}.zarr.zip"

    [[ -d "$zdir" ]] || { echo "WARNING – source missing: $zdir" >&2; continue; }
    if [[ -e "$zipout" ]]; then
        echo "✓ $zipout already exists – skipping"
        continue
    fi

    echo
    echo "> Zipping $slide   ($(sizeof "$zdir"))"
    start=$(date +%s)
    total_files=$(find "$zdir" -type f | wc -l | tr -d ' ')
    (( total_files == 0 )) && total_files=1   # avoid div/0 for empty dirs

    # progress log for “adding:” lines
    progress_log=$(mktemp)

    (
        cd "$(dirname "$zdir")"
        # -r recursive   -T test after write   (NO -q so we get “adding:” lines)
        zip -rT "$zipout" "$(basename "$zdir")" 2>&1 | tee "$progress_log" >/dev/null
    ) &
    zpid=$!

    spinner "$zpid" "$progress_log" "$total_files" "$start"
    wait "$zpid"; status=$?
    end=$(date +%s)

    if (( status == 0 )); then
        printf "\r✓ done in %s\n" "$(secs_to_hms $(( end - start )))"
    else
        echo -e "\nERROR – zip exited with status $status for $slide" >&2
        continue
    fi

    # ----- integrity & size checks ------------------------------------------
    if ! unzip -tq "$zipout" &>/dev/null; then
        echo "WARNING – integrity check FAILED for $zipout" >&2
    fi
    bytes=$(stat -c%s "$zipout" 2>/dev/null || stat -f%z "$zipout")
    if (( bytes > 2000000000000 )); then
        echo "ERROR – $zipout larger than 2 TB; split it!" >&2
    elif (( bytes > 100000000000 )); then
        echo "WARNING – $zipout is >100 GB ($(sizeof "$zipout"))" >&2
    fi
done
echo
echo "All SpatialData archives prepared."