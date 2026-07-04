import os
import re
import sys
import csv
from typing import Any, Dict, Optional
from tqdm import tqdm
from unsloth import FastLanguageModel, is_bfloat16_supported
import torch
import pandas as pd
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import subprocess
import time

# Global FLAGS dictionary
FLAGS: Dict[str, Any] = {}

def timed_step(step_name: str, step_emoji: str, step_number: int, func, *args, **kwargs):
    print(f"{step_emoji} Step {step_number}: {step_name}")
    step_start = time.time()
    
    try:
        result = func(*args, **kwargs)
        step_time = time.time() - step_start
        print(f"⏱️  Step {step_number} completed in {step_time:.2f} seconds")
        print("-" * 40)
        return result
    except Exception as e:
        step_time = time.time() - step_start
        print(f"❌ Step {step_number} failed after {step_time:.2f} seconds: {str(e)}")
        print("-" * 40)
        raise

def show_gpu_stats():
    # @title Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        try:
            print(subprocess.check_output(["nvidia-smi", "-L"], text=True))
        except Exception as e:
            print("nvidia-smi not available:", e)
    else:
        print("No GPU visible to PyTorch on this machine.")


    # 1) Check memory reported by PyTorch (Unsloth often relies on this)
    free, total = torch.cuda.mem_get_info(0)
    print("CUDA mem (GB) -> free:", round(free/1e9, 2), " total:", round(total/1e9, 2))

    # 2) Optional: nvidia-smi for human view
    print(subprocess.check_output(["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv,noheader"], text=True))

def load_llm_model(model_path, max_seq_length, dtype, load_4bit):
    global FLAGS
    print("Loading LLM model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        local_files_only=True,
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_4bit,
        full_finetuning = False,
        trust_remote_code = True,
        #device_map={"": 0},              # force all modules to cuda:0
        #gpu_memory_utilization=0.7,      # optional, Unsloth hinting
        #token = FLAGS["HF_ACCESS_TOKEN"] # "hf_...",
    )

    print("LLM model loaded from pretrained!")
    return model, tokenizer

def add_lora_adapters(llm_model, tokenizer):
    global FLAGS

    if FLAGS["mode"] == "finetune":
        print("Adding LoRA adapters...")

        """We now add LoRA adapters so we only need to update 1 to 10% of all parameters!"""
        model = FastLanguageModel.get_peft_model(
            llm_model,
            r = 8, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                              "gate_proj", "up_proj", "down_proj",],
            lora_alpha = 16,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
        print("Model finetuned with LoRA adapters!")
        return model, tokenizer
    else:
        # NOT finetuning mode, so no LoRA adapters added.
        return llm_model, tokenizer

def prepare_data(tokenizer, train_path, test_path):
    global FLAGS
    # Determine paths
    assert train_path and os.path.exists(train_path), f"Prepared train CSV not found: {train_path}"
    assert test_path and os.path.exists(test_path),   f"Prepared test CSV not found: {test_path}"

    print(f"✅ Using prepared instruction datasets:")
    print(f"  • Train: {train_path}")
    # Use engine='python' for better handling of multi-line fields and malformed CSV
    data_df_train = pd.read_csv(train_path, encoding='utf-8', engine='python', quoting=csv.QUOTE_MINIMAL, on_bad_lines='warn')
    print(f"  • Test:  {test_path}")
    data_df_test  = pd.read_csv(test_path, encoding='utf-8', engine='python', quoting=csv.QUOTE_MINIMAL, on_bad_lines='warn')

    # Minimal validation
    required_cols = {"instruction", "input", "output", "direction", "project"}
    missing_train = required_cols - set(data_df_train.columns)
    missing_test = required_cols - set(data_df_test.columns)
    assert not missing_train, f"Prepared train CSV missing columns: {missing_train}"
    assert not missing_test,  f"Prepared test CSV missing columns: {missing_test}"

    print("Data check complete!")

    # Build train/test datasets from v2 splits (no random split)
    print("🔄 Building training and test datasets...")

    # data check and transformation for the training

    test_df  = data_df_test.copy()
    print(f"🧪 Test examples: {len(test_df)} (projects: {test_df['project'].nunique()})")

    if(FLAGS["mode"] == "finetune"):
        # Keep all columns in your base DataFrame
        full_train_df = data_df_train.copy()

        # ---- Stats (use the full frames that still have 'project') ----
        print(f"📚 Training examples: {len(full_train_df)} (projects: {full_train_df['project'].nunique()})")

        # Ensure the three required columns exist and are strings
        needed = ["instruction", "input", "output"]
        assert all(c in full_train_df.columns for c in needed)

        # Split into what the model needs for training
        train_df = full_train_df[needed].fillna("").astype(str)

        # Convert to a Hugging Face dataset for training
        train_ds = Dataset.from_pandas(train_df, preserve_index=False)

        print("Final training dataset preparation in progress....")

        def formatting_training_prompts(examples):
            instructions = examples["instruction"]
            inputs       = examples["input"]
            outputs      = examples["output"]
            texts = []
            for instruction, input, output in zip(instructions, inputs, outputs):
                input_yaml = extract_clean_source_yaml(input)
                output_yaml = extract_clean_source_yaml(output)
                if FLAGS["migration_type"] == "gha_to_travis":
                    minimal_instruction = "Migrate this GitHub Actions workflow to Travis CI:"
                else:
                    minimal_instruction = "Migrate this Travis CI configuration to GitHub Actions:"
                
                messages = [
                    {"role": "user", "content": f"{minimal_instruction}\n{input_yaml}"},
                    {"role": "assistant", "content": output_yaml}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                texts.append(text)
            print("--------------------------------")
            lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"][0]) for text in texts]
            print(f"Avg # of tokens: {sum(lengths)/len(lengths):.0f}, Max # of tokens: {max(lengths)}")
            print("--------------------------------")
            return {"text": texts}

        training_df = train_ds.map(formatting_training_prompts, batched = True,)

        print("Final training and testing dataset preparation finished!")

        return training_df, test_df
    else:
        # NOT finetuning mode, so no training dataset preparation.
        return data_df_train, test_df

def prepare_trainer(model, tokenizer, training_dataset, max_seq_length, results_dir, 
                   batch_size=2, accum_steps=4, warmup=0.05, epochs=5, wd=0.01):
    global FLAGS
    if FLAGS["mode"] == "finetune":
        # Set learning rate based on migration type, according to the best learning rate found in the hyperparameter search.
        if FLAGS["migration_type"] == "travis_to_gha":
            lr = 1e-4
        else: # gha_to_travis
            lr = 1e-3
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=training_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            packing=False,
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=accum_steps,
                warmup_ratio=warmup,
                num_train_epochs=epochs,
                learning_rate=lr,
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=wd,
                lr_scheduler_type="cosine",
                seed=3407,
                output_dir=results_dir + "/.temp_checkpoints",
                report_to="none",
            ),
        )
        print(f"Fine-tuning trainer prepared with: batch={batch_size}, accum={accum_steps}, "
              f"warmup={warmup}, epochs={epochs}, lr={lr}, wd={wd}")
        return trainer
    else:
        return None
                       
def perform_model_training(trainer):
    global FLAGS
    if(FLAGS["mode"] == "finetune"):
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        trainer_stats = trainer.train()
        print("Training completed!")

        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(
            f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
        )
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")
        print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
        print("Training completed!")

# Utility functions for code normalization and evaluation
def clean_yaml_output(raw_output):
    """
    Clean up LLM output to extract only pure YAML content.

    Some models may still include markdown formatting or explanatory text
    despite explicit prompts. This function removes common unwanted elements.

    Args:
        raw_output (str): Raw output from LLM

    Returns:
        str: Cleaned YAML content
    """
    if not raw_output:
        return ""

    # Remove common markdown code block patterns
    cleaned = raw_output.strip()

    # Remove opening markdown patterns
    patterns_to_remove_start = [
        "Here is the migrated GitHub Actions workflow file from your provided YAML:",
        "Here is the migrated GitHub Actions workflow file:",
        "Here is the migrated Travis CI configuration:",
        "This GitHub Actions workflow file maintains",
        "This Travis CI configuration",
        "Here is the migrated YAML:",
        "```yaml",
        "```yml",
        "```",
        "The migrated YAML file is:",
        "Migrated YAML:",
    ]

    for pattern in patterns_to_remove_start:
        if cleaned.startswith(pattern):
            cleaned = cleaned[len(pattern):].strip()

    # Split into lines to find YAML boundaries
    lines = cleaned.split('\n')
    yaml_start_idx = 0
    yaml_end_idx = len(lines)

    # Look for the actual YAML content start
    common_yaml_starts = ['name:', 'language:', 'on:', 'jobs:', 'version:', 'os:', 'dist:', 'sudo:', 'script:', 'install:', 'before_script:', 'after_script:']

    for i, line in enumerate(lines):
        line_stripped = line.strip().lower()
        if any(line_stripped.startswith(start) for start in common_yaml_starts):
            yaml_start_idx = i
            break

    # Find where YAML content ends (look for explanatory text that comes after YAML)
    explanatory_patterns = [
        "This GitHub Actions workflow file maintains",
        "This Travis CI configuration",
        "The migrated YAML file",
        "This workflow",
        "Note:",
        "Important:",
        "Here is the migrated",
        "The above",
        "This configuration"
    ]

    # Start looking for end patterns after we found the YAML start
    for i in range(yaml_start_idx + 1, len(lines)):
        line = lines[i].strip()

        # Stop at closing markdown
        if line == "```":
            yaml_end_idx = i
            break

        # Stop at explanatory text patterns
        if any(line.startswith(pattern) for pattern in explanatory_patterns):
            yaml_end_idx = i
            break

        # Stop at empty lines followed by explanatory text
        if line == "" and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if any(next_line.startswith(pattern) for pattern in explanatory_patterns):
                yaml_end_idx = i
                break

    # Extract only the YAML content (preserve model's exact output)
    yaml_content = '\n'.join(lines[yaml_start_idx:yaml_end_idx]).strip()

    # Final cleanup - remove any trailing markdown only if it's at the very end
    if yaml_content.endswith("```"):
        yaml_content = yaml_content[:-3].strip()

    return yaml_content

def extract_clean_source_yaml(raw_input):
    """
    Extract clean source YAML content by removing instruction prefixes

    Args:
        raw_input (str): Raw input text that may contain instruction prefixes

    Returns:
        str: Clean YAML content without instruction prefixes
    """
    if raw_input is None or pd.isna(raw_input):
        return ""

    # Remove known instruction prefixes
    prefixes_to_remove = [
        'Migrate this TRAVIS TO GHA configuration:\n\n',
        'Migrate this GHA TO TRAVIS configuration:\n\n',
        'Migrate this TRAVIS TO GHA configuration:',
        'Migrate this GHA TO TRAVIS configuration:'
    ]

    cleaned = raw_input.strip()
    for prefix in prefixes_to_remove:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    # Fallback: if we still have instructional text, try splitting on double newlines
    if cleaned.startswith('Migrate this') or 'configuration:' in cleaned[:50]:
        parts = cleaned.split('\n\n', 1)
        if len(parts) > 1:
            cleaned = parts[1].strip()

    # Replace secure tokens with placeholder (handles both quoted and unquoted formats)
    cleaned = re.sub(r'(secure:\s*)(\S+)', r'\1"********************"', cleaned)

    return cleaned

def build_dynamic_examples_few_shot_example_block(
    few_shot_pool_df: pd.DataFrame,
    exclude_index: Any,
    migration_type: str,
) -> str:
    """
    Build {{ FEW_SHOT_EXAMPLES }} content from up to three random rows in the
    instruction dataset, excluding the row being evaluated (exclude_index).
    """
    if few_shot_pool_df is None or len(few_shot_pool_df) == 0:
        return ""

    if exclude_index is not None:
        candidates = few_shot_pool_df.drop(index=exclude_index, errors="ignore")
    else:
        candidates = few_shot_pool_df
    if "direction" in candidates.columns:
        candidates = candidates[candidates["direction"] == migration_type]

    if len(candidates) == 0:
        return ""

    n_pick = min(3, len(candidates))
    sampled = candidates.sample(n=n_pick, replace=False)

    parts: list[str] = ["Here are some example migrations for reference:\n"]
    for k, (_, row) in enumerate(sampled.iterrows(), start=1):
        in_yaml = extract_clean_source_yaml(row.get("input", ""))
        out_yaml = extract_clean_source_yaml(row.get("output", ""))
        if migration_type == "travis_to_gha":
            parts.append(
                f"\n## Example {k}: Migration from Travis CI Configuration to GitHub Action Workflow\n"
                f"### Travis CI (.travis.yml) Configuration:\n```yaml\n{in_yaml}\n```\n\n"
                f"### Corresponding Migrated GitHub Actions Workflow:\n```yaml\n{out_yaml}\n```\n"
                f"---\n"
            )
        else:
            parts.append(
                f"\n## Example {k}: Migration from GitHub Action Workflow to Travis CI Configuration\n"
                f"### GitHub Actions Workflow:\n```yaml\n{in_yaml}\n```\n\n"
                f"### Corresponding Migrated Travis CI Configuration (.travis.yml):\n```yaml\n{out_yaml}\n```\n"
                f"---\n"
            )
    return "".join(parts)

def handle_zero_n_few_shot_prompts_format(
    cigrate_config_path: str,
    input_yaml: str,
    few_shot_pool_df: Optional[pd.DataFrame] = None,
    exclude_index: Any = None,
):
    global FLAGS
    # Read prompts from config folder
    base_config = os.path.join(cigrate_config_path, "prompts")
    def read_prompt(fname):
        fpath = os.path.join(base_config, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            return f.read()

    few_shot_fixed = FLAGS["mode"] == "few_shot"
    few_shot_dynamic_examples = FLAGS["mode"] == "few_shot_dynamic_examples"
    migration_type = FLAGS["migration_type"]
    # Configure conversion prompts based on target format
    # Each conversion type has specific system and user prompts for optimal results
    if migration_type == "travis_to_gha":
        system_prompt = read_prompt("travis_to_gha_system.txt")
        user_prompt_template = read_prompt("travis_to_gha_user.txt")
        example_block = ""
        if few_shot_fixed:
            # Use the entire example file content as the block
            example_file_path = os.path.join(cigrate_config_path, "few-shot-examples/travis_to_gha.txt")
            if os.path.exists(example_file_path):
                with open(example_file_path, "r", encoding="utf-8") as ef:
                    example_block = ef.read()
        elif few_shot_dynamic_examples:
            example_block = build_dynamic_examples_few_shot_example_block(
                few_shot_pool_df, exclude_index, migration_type
            )
        user_prompt = user_prompt_template.replace("{{ FEW_SHOT_EXAMPLES }}", example_block).replace("{{ INPUT_CODE }}", input_yaml)
    elif migration_type == "gha_to_travis":
        system_prompt = read_prompt("gha_to_travis_system.txt")
        user_prompt_template = read_prompt("gha_to_travis_user.txt")
        example_block = ""
        if few_shot_fixed:
            example_file_path = os.path.join(cigrate_config_path, "few-shot-examples/gha_to_travis.txt")
            if os.path.exists(example_file_path):
                with open(example_file_path, "r", encoding="utf-8") as ef:
                    example_block = ef.read()
        elif few_shot_dynamic_examples:
            example_block = build_dynamic_examples_few_shot_example_block(
                few_shot_pool_df, exclude_index, migration_type
            )
        user_prompt = user_prompt_template.replace("{{ FEW_SHOT_EXAMPLES }}", example_block).replace("{{ INPUT_CODE }}", input_yaml)
    else:
        raise ValueError("❌ Invalid conversion type. Use 'travis_to_gha' or 'gha_to_travis'.")

    return system_prompt, user_prompt

def generate_migration(
    model,
    tokenizer,
    cigrate_config_path,
    input_yaml,
    few_shot_pool_df: Optional[pd.DataFrame] = None,
    exclude_index: Any = None,
):
    global FLAGS
    if FLAGS["mode"] == "finetune":
        # Fine-tuned: minimal instruction
        if FLAGS["migration_type"] == "gha_to_travis":
            instruction = "Migrate this GitHub Actions workflow to Travis CI:"
        else:
            instruction = "Migrate this Travis CI configuration to GitHub Actions:"
        full_prompt_content = f"{instruction}\n{input_yaml}"
    else:
        # Zero-shot or Few-shot: use original prompts
        system_prompt, user_prompt = handle_zero_n_few_shot_prompts_format(
            cigrate_config_path, input_yaml, few_shot_pool_df, exclude_index
        )
        full_prompt_content = f"{system_prompt}\n\n{user_prompt}"

    # Apply template WITH generation prompt this time
    messages = [{"role": "user", "content": full_prompt_content}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    # Estimate max_new_tokens based on input length
    max_new_tokens = min(max(len(input_yaml.split()) * 2, 512), 1024)  # 2x input, capped at 1024

    FastLanguageModel.for_inference(model)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache = True)

    # decode only new tokens (not the prompt)
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = outputs[:, prompt_len:]
    model_response = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

    # Clean code fences if present
    generated_yaml = re.sub(r'^```(yaml|yml)?\s*', '', model_response, flags=re.IGNORECASE)
    generated_yaml = re.sub(r'\s*```$', '', generated_yaml)
    return generated_yaml.strip()

def write_project_outputs(short_model_name, project_name, outputs, results_dir):
    project_dir = f"{results_dir}/{project_name}"
    os.makedirs(project_dir, exist_ok=True)
    # Ensure subdirs exist
    subdirs = [
        "01_Original_Travis",
        "02_Original_GHA",
        f"03_CIgrate_Travis_to_GHA/{short_model_name}",
        f"04_CIgrate_GHA_to_Travis/{short_model_name}"
    ]
    for subdir in subdirs:
        os.makedirs(f"{project_dir}/{subdir}", exist_ok=True)

    # Write originals if available
    if outputs.get('travis_original'):
        with open(f"{project_dir}/01_Original_Travis/travis.yml", 'w') as f:
            f.write(outputs['travis_original'])
    if outputs.get('gha_original'):
        with open(f"{project_dir}/02_Original_GHA/actions.yml", 'w') as f:
            f.write(outputs['gha_original'])

    # Write generated migrations if available
    if outputs.get('travis_to_gha_generated'):
        with open(f"{project_dir}/03_CIgrate_Travis_to_GHA/{short_model_name}/actions.yml", 'w') as f:
            f.write(outputs['travis_to_gha_generated'])
    if outputs.get('gha_to_travis_generated'):
        with open(f"{project_dir}/04_CIgrate_GHA_to_Travis/{short_model_name}/travis.yml", 'w') as f:
            f.write(outputs['gha_to_travis_generated'])

def evaluate_model(short_model_name, model, tokenizer, test_df, results_dir, cigrate_config_path):
    # Fine-tuned model evaluation with bidirectional testing + writing migrations files
    print("🔄 Evaluating fine-tuned model with bidirectional testing (writing migrations files)...")

    # Determine which projects to evaluate
    selected_projects = list(test_df['project'].unique())
    print(f"📊 Evaluating all {len(selected_projects)} projects")

    # Build an index list of test_df rows belonging to the selected projects
    selected_indices = test_df.index[test_df['project'].isin(selected_projects)].tolist()
    print(f"📄 Selected test rows: {len(selected_indices)}")

    print(f"📊 Testing fine-tuned model on {len(selected_indices)} samples across {len(selected_projects)} projects...")

    project_outputs = {}
    for j, i in enumerate(tqdm(selected_indices, desc=f"Generating a {FLAGS['migration_type'].replace('_to_', '-to-')} migration")):
        # Get test data
        test_row = test_df.loc[i]
        project_name = test_row['project']
        direction = test_row['direction']
        instruction = test_row['instruction']
        input_yaml = test_row['input']
        ground_truth = test_row['output']

        subdir = "03_CIgrate_Travis_to_GHA" if direction == "travis_to_gha" else "04_CIgrate_GHA_to_Travis"
        out_name = "actions.yml" if direction == "travis_to_gha" else "travis.yml"
        output_yml_path = f"{results_dir}/{project_name}/{subdir}/{short_model_name}/{out_name}"

        # Skip if the output YAML file already exists
        if os.path.exists(output_yml_path):
            continue

        # Extract clean source YAML
        source_yaml = extract_clean_source_yaml(input_yaml)
        ground_truth_yaml = extract_clean_source_yaml(ground_truth)

        # Generate migration and clean
        clean_generated_yaml = generate_migration(
            model,
            tokenizer,
            cigrate_config_path,
            source_yaml,
            few_shot_pool_df=test_df if FLAGS["mode"] == "few_shot_dynamic_examples" else None,
            exclude_index=i if FLAGS["mode"] == "few_shot_dynamic_examples" else None,
        )

        if project_name not in project_outputs:
            project_outputs[project_name] = {
                'travis_original': None,
                'gha_original': None,
                'travis_to_gha_generated': None,
                'gha_to_travis_generated': None
            }

        if direction == 'travis_to_gha':
            project_outputs[project_name] = {
                'travis_original': source_yaml,
                'gha_original': ground_truth_yaml,
                'travis_to_gha_generated': clean_generated_yaml,
                'gha_to_travis_generated': None
            }
        else:  # gha_to_travis
            project_outputs[project_name] = {
                'travis_original': ground_truth_yaml,
                'gha_original': source_yaml,
                'travis_to_gha_generated': None,
                'gha_to_travis_generated': clean_generated_yaml
            }
        
        # Write this project's current state after each sample
        write_project_outputs(short_model_name, project_name, project_outputs[project_name], results_dir)

    print(f"✅ Migrations files written under: {results_dir}/")
    print(f"✅ Generated migrations for {len(project_outputs)} unique projects")

def save_finetuned_model_full(model, tokenizer, saved_model_path):
    global FLAGS
    if FLAGS["mode"] == "finetune":
        saved_model_specific_path = f"{saved_model_path}/finetuned_model_full_{FLAGS['model_friendly_name']}_{FLAGS['migration_type']}"
        FastLanguageModel.for_inference(model)
        model.save_pretrained_merged(saved_model_specific_path, tokenizer, save_method = "merged_16bit")

def save_finetuned_model_lora_adapters_only(model, tokenizer, saved_model_path):
    global FLAGS
    if FLAGS["mode"] == "finetune":
        saved_model_specific_path = f"{saved_model_path}/finetuned_model_lora_adapters_only_{FLAGS['model_friendly_name']}_{FLAGS['migration_type']}"
        FastLanguageModel.for_inference(model)
        model.save_pretrained(saved_model_specific_path, safe_serialization=True)
        tokenizer.save_pretrained(saved_model_specific_path)

def run_cigrate_steps(short_model_name, model_path, max_seq_length, dtype, load_4bit, train_path, test_path, results_dir, cigrate_config_path, saved_model_path):
    show_gpu_stats()

    llm_model, tokenizer = timed_step("Model Loading", "🤖", 1, load_llm_model, model_path, max_seq_length, dtype, load_4bit)
    
    model, tokenizer = timed_step("LoRA Adapters", "🤖", 2, add_lora_adapters, llm_model, tokenizer)
    
    training_dataset, test_df = timed_step("Data Preparation", "📚", 3, prepare_data, tokenizer, train_path, test_path)
    
    trainer = timed_step("Trainer Preparation", "🏋️", 4, prepare_trainer, model, tokenizer, training_dataset, max_seq_length, results_dir)
    
    timed_step("Model Training", "🎯", 5, perform_model_training, trainer)
    
    timed_step("Model Evaluation", "🧪", 6, evaluate_model, short_model_name, model, tokenizer, test_df, results_dir, cigrate_config_path)
    
    timed_step("Saving Full Fine-tuned Model", "💾", 7, save_finetuned_model_full, model, tokenizer, saved_model_path)
    timed_step("Saving Fine-tuned Model LoRA Adapters Only", "💾", 8, save_finetuned_model_lora_adapters_only, model, tokenizer, saved_model_path)
 
if __name__ == "__main__":
    model_dir = sys.argv[1]
    model_name = sys.argv[2]
    max_seq_length = int(sys.argv[3])
    load_4bit = sys.argv[4] == 'True'
    cigrate_config_path = sys.argv[5]
    input_data_path = sys.argv[6]
    output_data_path = sys.argv[7]
    migration_type = sys.argv[8]
    mode = sys.argv[9]
    dtype = None # None for auto detection

    # Start overall timing
    total_start_time = time.time()
    print("=" * 87)
    print("🚀 Starting Unsloth fine-tuning pipeline for model: ", model_name)
    print("=" * 87)
    
    model_name_mapping = {
        "meta-llama/Llama-3.1-8B-Instruct" : "llama3_1_8b",
        "google/gemma-3-12b-it" : "gemma3_12b",
        "google/gemma-3-4b-it" : "gemma3_4b",
        "mistralai/Mistral-7B-Instruct-v0.3" : "mistral_7b",
        "codellama/CodeLlama-7b-Instruct-hf" : "codellama_7b",
        "google/codegemma-7b-it" : "codegemma_7b",
    }

    # Friendly model name for output directories and file organization
    short_model_name = model_name_mapping[model_name]
    MODEL_FRIENDLY_NAME = model_name.split("/")[-1].replace("-", "_").replace(".", "_")
    results_dir = f"{output_data_path}"
    saved_model_path = f"{results_dir}/.saved_models/"

    os.makedirs(results_dir, exist_ok=True)

    # Populate global FLAGS
    FLAGS.update({
        "mode": mode, # "finetune" or "zero_shot" or "few_shot" or "few_shot_dynamic_examples" or "just_generate"
        "migration_type": migration_type, # "gha_to_travis" or "travis_to_gha"
        "model_friendly_name": MODEL_FRIENDLY_NAME
    })

    # Resolve prepared paths
    if FLAGS["mode"] == "finetune":
        train_path = os.path.join(input_data_path, f"instruction_dataset_train_{FLAGS['migration_type']}.csv") # 'instruction_dataset_train_travis_to_gha.csv'
        test_path  = os.path.join(input_data_path, f"instruction_dataset_test_{FLAGS['migration_type']}.csv") # 'instruction_dataset_test_travis_to_gha.csv'
    else:
        # Pointing to the same file for train and test, as no training is needed in zero-shot, few-shot, and few-shot-dynamic-examples modes.
        train_path = os.path.join(input_data_path, f"instruction_dataset_{FLAGS['migration_type']}.csv") # 'instruction_dataset_travis_to_gha.csv'
        test_path  = os.path.join(input_data_path, f"instruction_dataset_{FLAGS['migration_type']}.csv") # 'instruction_dataset_travis_to_gha.csv'

    if FLAGS["mode"] == "just_generate": # Use the saved model for generation
        if not os.path.exists(saved_model_path):
            raise FileNotFoundError(f"Saved fine-tuned model not found at: {saved_model_path}")
        model_path = saved_model_path
        print(f"Using saved fine-tuned model at: {model_path}")
    else: # Use the base model for training
        model_path = f"{model_dir}/{model_name.replace('/', '_').replace('.', '-')}"
        print(f"Using base model at: {model_path}")

    run_cigrate_steps(short_model_name, model_path, max_seq_length, dtype, load_4bit, train_path, test_path, results_dir, cigrate_config_path, saved_model_path)
    
    # Final timing summary
    total_time = time.time() - total_start_time
    print("=" * 60)
    print("🎉 All done!")
    print(f"⏱️  Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("=" * 60)