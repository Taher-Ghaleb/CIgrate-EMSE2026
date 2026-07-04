#!/bin/bash
echo "Run,Mode,Migration Direction,Model,Step 1: Model Loading,Step 2: LoRA Adapters,Step 3: Data Preparation,Step 4: Trainer Preparation,Step 5: Model Training,Step 6: Model Evaluation,Step 7: Model Saving,Total Seconds,Total Minutes" > CIgrate_Execution_Time_Results.csv

paths=(
    "/home/tghaleb/CI_LLM/slurm_out/NazmulWork-FinalHyperparameters/Sample_153/Set_90_10"
    "/home/tghaleb/CI_LLM/slurm_out/NazmulWork-FinalHyperparameters/Sample_153/Set_80_20"
    "/home/tghaleb/CI_LLM/slurm_out/NazmulWork-FinalHyperparameters/Sample_153/Set_Full"
)

for path in "${paths[@]}"; do
    find "$path" -type f -name "*.out" | while read -r f; do
        
        run=$(basename "$(dirname "$f")")
        filename=$(basename "$f" .out)

        # 1️⃣ Extract migration direction
        migration_direction=$(echo "$filename" | grep -oE 'gha_to_travis|travis_to_gha')
        if [[ -z "$migration_direction" ]]; then
            echo "Skipping (no migration found): $filename" >&2
            continue
        fi

        # 2️⃣ Mode = everything before migration direction
        mode="${filename%%$migration_direction*}"
        mode=$(echo "$mode" | sed 's/_$//')  # remove trailing underscore if exists

        # 3️⃣ Everything after migration direction
        after="${filename#*$migration_direction}"
        after="${after#_}"  # remove leading underscore

        # 4️⃣ Remove optional trailing _number
        model_name=$(echo "$after" | sed -E 's/_[0-9]+$//')

        # 5️⃣ Extract step times (1–7)
        for i in {1..7}; do
            step_time=$(grep "Step $i completed in" "$f" | sed -E "s/.*Step $i completed in ([0-9]+\.[0-9]+) seconds.*/\1/")
            eval "step$i=${step_time:-0}"
        done

        # 6️⃣ Extract total execution time (robust to emojis or extra words)
        total_line=$(grep "Total execution time" "$f")

        total_seconds=$(echo "$total_line" | sed -E 's/.*Total execution time[^:]*: ([0-9]+\.[0-9]+) seconds.*/\1/')
        total_minutes=$(echo "$total_line" | sed -E 's/.*\(([0-9]+\.[0-9]+) minutes\).*/\1/')

        # 7️⃣ Write CSV row
        if [[ -n "$total_seconds" && -n "$total_minutes" ]]; then
            echo "$run,$mode,$migration_direction,$model_name,$step1,$step2,$step3,$step4,$step5,$step6,$step7,$total_seconds,$total_minutes" \
                >> CIgrate_Execution_Time_Results-FinalHyperparameters.csv
        else
            echo "⚠️ Missing total time for file: $f" >&2
        fi

    done
done
