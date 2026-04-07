#!/bin/bash
# Batch evaluation for Motion324: GT pipeline + Video-mesh pipeline
# Usage: bash scripts/batch_eval_all.sh [--release_dir PATH] [--gt_only] [--video_only] [--skip_inference] [--skip_eval]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

RELEASE_DIR="$PROJECT_ROOT/examples/release_80"
if [ "${1:-}" == "--release_dir" ]; then
    RELEASE_DIR="$2"
    shift 2
fi
CKPT="$PROJECT_ROOT/experiments/checkpoints/ckpt_0000000000060000.pt"
CONFIG="$PROJECT_ROOT/configs/dyscene.yaml"
SHORT_LIST="$PROJECT_ROOT/dataset/short_videos.txt"
LONG_LIST="$PROJECT_ROOT/dataset/long_videos.txt"
GT_OUTPUT_BASE="$PROJECT_ROOT/experiments/batch_eval_gt"
VIDEO_OUTPUT_BASE="$PROJECT_ROOT/experiments/batch_eval_generated"
RUN_GT=true
RUN_VIDEO=true
SKIP_INFERENCE=false
SKIP_EVAL=false

for arg in "$@"; do
    case "$arg" in
        --gt_only)        RUN_VIDEO=false ;;
        --video_only)     RUN_GT=false ;;
        --skip_inference) SKIP_INFERENCE=true ;;
        --skip_eval)      SKIP_EVAL=true ;;
        *)                echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

safe_name() { echo "$1" | sed 's/ /_/g; s/|/_/g'; }
read_list() {
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" ]] && continue
        echo "$line"
    done < "$1"
}

run_gt_pipeline() {
    local name="$1"
    local sname; sname="$(safe_name "$name")"
    local pcd_dir="$RELEASE_DIR/pcds/${name}_pointclouds"
    local glb_path="$RELEASE_DIR/glbs/${name}.glb"
    local image_dir="$RELEASE_DIR/images/${name}_images/camera_0"
    local out_dir="$GT_OUTPUT_BASE/$sname"
    local pred_path="$out_dir/output_animation.glb"
    local gt_dir="$pcd_dir"
    local log_dir="$GT_OUTPUT_BASE/logs"
    mkdir -p "$log_dir"
    if [ ! -d "$pcd_dir" ]; then echo "  [GT] SKIP: pcd dir not found: $pcd_dir"; return 1; fi
    if [ ! -d "$image_dir" ]; then echo "  [GT] SKIP: image dir not found: $image_dir"; return 1; fi
    if [ "$SKIP_INFERENCE" = false ]; then
        if [ ! -f "$pred_path" ]; then
            echo "  [GT] Running inference..."
            mkdir -p "$out_dir"
            local pcd_arg="$pcd_dir" img_arg="$image_dir" glb_arg="$glb_path" created_links=()
            if [[ "$pcd_dir" == *" "* ]] || [[ "$pcd_dir" == *"|"* ]]; then
                local pcd_link="$out_dir/input_pcds"; rm -f "$pcd_link"
                ln -s "$(readlink -f "$pcd_dir")" "$pcd_link"
                pcd_arg="$pcd_link"; created_links+=("$pcd_link")
            fi
            if [[ "$image_dir" == *" "* ]] || [[ "$image_dir" == *"|"* ]]; then
                local img_link="$out_dir/input_images"; rm -f "$img_link"
                ln -s "$(readlink -f "$image_dir")" "$img_link"
                img_arg="$img_link"; created_links+=("$img_link")
            fi
            if [[ "$glb_path" == *" "* ]] || [[ "$glb_path" == *"|"* ]]; then
                local glb_link="$out_dir/input_mesh.glb"; rm -f "$glb_link"
                ln -s "$(readlink -f "$glb_path")" "$glb_link"
                glb_arg="$glb_link"; created_links+=("$glb_link")
            fi
            python "$PROJECT_ROOT/scripts/inference_with_gt.py" \
                --config="$CONFIG" \
                training.resume_ckpt="$CKPT" \
                model.class_name=model.Pcd_motion.Motion_Latent_Model \
                training.num_shape_samples=16384 \
                training.frames=256 \
                start_frame=0 \
                use_segmentation=False \
                data_dir="$pcd_arg" \
                glb_path="$glb_arg" \
                video_path="$img_arg" \
                output_dir="$out_dir" \
                2>&1 | tee "$log_dir/${sname}_inference.log"
            local rc=${PIPESTATUS[0]}
            for lnk in "${created_links[@]}"; do rm -f "$lnk"; done
            if [ "$rc" -ne 0 ]; then echo "  [GT] Inference FAILED (rc=$rc)"; return 1; fi
        else
            echo "  [GT] Inference output exists, skipping"
        fi
    fi
    if [ "$SKIP_EVAL" = false ]; then
        if [ -f "$pred_path" ]; then
            echo "  [GT] Running evaluation..."
            python "$PROJECT_ROOT/evaluation/evaluation_pcd.py" \
                --gt_path "$gt_dir" --pred_path "$pred_path" --icp_viz \
                2>&1 | tee "$log_dir/${sname}_eval.log"
        else
            echo "  [GT] No prediction found, skipping eval"; return 1
        fi
    fi
    return 0
}

run_video_pipeline() {
    local name="$1"
    local sname; sname="$(safe_name "$name")"
    local image_dir="$RELEASE_DIR/images/${name}_images/camera_0"
    local out_dir="$VIDEO_OUTPUT_BASE/$sname"
    local gt_dir="$RELEASE_DIR/pcds/${name}_pointclouds"
    local log_dir="$VIDEO_OUTPUT_BASE/logs"
    mkdir -p "$log_dir" "$out_dir"
    if [ ! -d "$image_dir" ]; then echo "  [VIDEO] SKIP: image dir not found: $image_dir"; return 1; fi

    # Determine video basename for consistent path computation
    local video_basename="camera_0"
    if [[ "$image_dir" == *" "* ]] || [[ "$image_dir" == *"|"* ]]; then
        video_basename="input_images"
    fi
    local video_mp4="$out_dir/${video_basename}.mp4"
    # 4D_from_video.sh outputs to <video_dir>/<video_name>_processed/animation/
    local pred_path="$out_dir/${video_basename}_processed/animation/output_animation.fbx"

    if [ "$SKIP_INFERENCE" = false ]; then
        if [ ! -f "$pred_path" ]; then
            local created_links=()

            # Symlink image dir if name contains spaces/pipes
            local img_arg="$image_dir"
            if [[ "$image_dir" == *" "* ]] || [[ "$image_dir" == *"|"* ]]; then
                local img_link="$out_dir/input_images"; rm -f "$img_link"
                ln -s "$(readlink -f "$image_dir")" "$img_link"
                img_arg="$img_link"; created_links+=("$img_link")
            fi

            # Step 1: Concatenate images into video
            echo "  [VIDEO] Step 1: Converting images to video..."
            python "$PROJECT_ROOT/scripts/images2video.py" \
                "$img_arg" "$out_dir" \
                2>&1 | tee "$log_dir/${sname}_images2video.log"
            if [ ! -f "$video_mp4" ]; then
                echo "  [VIDEO] FAILED: Video not created at $video_mp4"
                for lnk in "${created_links[@]}"; do rm -f "$lnk"; done
                return 1
            fi

            # Step 2: Run 4D_from_video.sh (video -> rmbg + mesh generation + animation)
            echo "  [VIDEO] Step 2: Running 4D_from_video.sh..."
            bash "$PROJECT_ROOT/scripts/4D_from_video.sh" \
                "$video_mp4" \
                2>&1 | tee "$log_dir/${sname}_inference.log"
            local rc=${PIPESTATUS[0]}
            for lnk in "${created_links[@]}"; do rm -f "$lnk"; done
            if [ "$rc" -ne 0 ]; then echo "  [VIDEO] Inference FAILED (rc=$rc)"; return 1; fi
        else
            echo "  [VIDEO] Inference output exists, skipping"
        fi
    fi
    if [ "$SKIP_EVAL" = false ]; then
        if [ -f "$pred_path" ]; then
            echo "  [VIDEO] Running evaluation..."
            python "$PROJECT_ROOT/evaluation/evaluation_pcd.py" \
                --gt_path "$gt_dir" --pred_path "$pred_path" --icp_viz \
                2>&1 | tee "$log_dir/${sname}_eval.log"
        else
            echo "  [VIDEO] No prediction found, skipping eval"; return 1
        fi
    fi
    return 0
}

parse_results() {
    local txt_path="$1"
    if [ -f "$txt_path" ]; then
        local cd_val fs_val
        cd_val=$(grep "^cd_mean_" "$txt_path" 2>/dev/null | sed 's/cd_mean_//')
        fs_val=$(grep "^fs_mean_" "$txt_path" 2>/dev/null | sed 's/fs_mean_//')
        echo "${cd_val:-FAIL} ${fs_val:-FAIL}"
    else
        echo "FAIL FAIL"
    fi
}

print_summary() {
    local title="$1" split_label="$2"; shift 2; local names=("$@")
    echo ""
    echo "================================================================"
    echo "  $title  ($split_label, ${#names[@]} samples)"
    echo "================================================================"
    printf "%-70s  %12s  %12s\n" "Sample" "CD mean" "F-score"
    echo "------------------------------------------------------------------------"
    local cd_sum=0 fs_sum=0 count=0
    for name in "${names[@]}"; do
        local gt_dir="$RELEASE_DIR/pcds/${name}_pointclouds"
        local result_txt="$gt_dir/evaluation_results.txt"
        local vals; vals=$(parse_results "$result_txt")
        local cd_val fs_val
        cd_val=$(echo "$vals" | awk '{print $1}')
        fs_val=$(echo "$vals" | awk '{print $2}')
        if [ "$cd_val" != "FAIL" ] && [ -n "$cd_val" ]; then
            printf "%-70s  %12s  %12s\n" "${name:0:70}" "$cd_val" "$fs_val"
            cd_sum=$(python3 -c "print($cd_sum + $cd_val)")
            fs_sum=$(python3 -c "print($fs_sum + $fs_val)")
            count=$((count + 1))
        else
            printf "%-70s  %12s  %12s\n" "${name:0:70}" "FAILED" "FAILED"
        fi
    done
    echo "------------------------------------------------------------------------"
    if [ "$count" -gt 0 ]; then
        local cd_mean fs_mean
        cd_mean=$(python3 -c "print(f'{$cd_sum / $count:.6f}')")
        fs_mean=$(python3 -c "print(f'{$fs_sum / $count:.6f}')")
        printf "%-70s  %12s  %12s\n" "MEAN" "$cd_mean" "$fs_mean"
    fi
    printf "%-70s  %12s\n" "SUCCESS" "$count / ${#names[@]}"
    echo "================================================================"
}

echo "=========================================="
echo "  Motion324 Batch Evaluation"
echo "=========================================="
echo "Release dir  : $RELEASE_DIR"
echo "GT output    : $GT_OUTPUT_BASE"
echo "Video output : $VIDEO_OUTPUT_BASE"
echo "Run GT       : $RUN_GT"
echo "Run Video    : $RUN_VIDEO"
echo "Skip inference: $SKIP_INFERENCE"
echo "Skip eval     : $SKIP_EVAL"
echo "=========================================="

mapfile -t SHORT_NAMES < <(read_list "$SHORT_LIST")
mapfile -t LONG_NAMES < <(read_list "$LONG_LIST")
ALL_NAMES=("${SHORT_NAMES[@]}" "${LONG_NAMES[@]}")
TOTAL=${#ALL_NAMES[@]}
echo "Samples: ${#SHORT_NAMES[@]} short + ${#LONG_NAMES[@]} long = $TOTAL total"

if [ "$RUN_GT" = true ]; then
    echo ""; echo "============================================="
    echo "  GT Pipeline (inference_with_gt.py)"
    echo "============================================="
    mkdir -p "$GT_OUTPUT_BASE"
    gt_i=0; gt_ok=0; gt_fail=0
    for name in "${ALL_NAMES[@]}"; do
        gt_i=$((gt_i + 1)); echo ""; echo "[$gt_i/$TOTAL] $name"
        if run_gt_pipeline "$name"; then gt_ok=$((gt_ok + 1)); else gt_fail=$((gt_fail + 1)); fi
    done
    echo "GT Pipeline done: $gt_ok succeeded, $gt_fail failed out of $TOTAL"
    print_summary "GT Pipeline" "Short Videos" "${SHORT_NAMES[@]}"
    print_summary "GT Pipeline" "Long Videos"  "${LONG_NAMES[@]}"
fi

if [ "$RUN_VIDEO" = true ]; then
    echo ""; echo "============================================="
    echo "  Video-Mesh Pipeline (inference_with_video_mesh.py)"
    echo "============================================="
    mkdir -p "$VIDEO_OUTPUT_BASE"
    vid_i=0; vid_ok=0; vid_fail=0
    for name in "${ALL_NAMES[@]}"; do
        vid_i=$((vid_i + 1)); echo ""; echo "[$vid_i/$TOTAL] $name"
        if run_video_pipeline "$name"; then vid_ok=$((vid_ok + 1)); else vid_fail=$((vid_fail + 1)); fi
    done
    echo "Video Pipeline done: $vid_ok succeeded, $vid_fail failed out of $TOTAL"
    print_summary "Video-Mesh Pipeline" "Short Videos" "${SHORT_NAMES[@]}"
    print_summary "Video-Mesh Pipeline" "Long Videos"  "${LONG_NAMES[@]}"
fi

echo ""; echo "=========================================="
echo "  All evaluations complete!"
echo "=========================================="
