import os
import re
import sys
from typing import Dict, Any
from tqdm import tqdm
import pandas as pd
from datasets import Dataset
import time
from openai import OpenAI

# To override Python's built-in print function to write output to both the console and a log file
class Logger:
    def __init__(self, log_file):
        self.log_file = log_file

    def write(self, message):
        with open(self.log_file, 'a') as f:
            f.write(message)
        sys.__stdout__.write(message)  # Print to console as well

    def flush(self):
        sys.__stdout__.flush()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "<your_openai_api_key_here>")

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

def prepare_data(test_path):
    # Determine paths
    assert test_path and os.path.exists(test_path),   f"Prepared test CSV not found: {test_path}"

    instruction_df_test  = pd.read_csv(test_path, encoding='utf-8')

    # Minimal validation
    required_cols = {"instruction", "input", "output", "direction", "project"}
    missing_test = required_cols - set(instruction_df_test.columns)
    assert not missing_test,  f"Prepared test CSV missing columns: {missing_test}"

    print("Data check complete!")

    test_df  = instruction_df_test.copy()
    print(f"🧪 Test examples: {len(test_df)} (projects: {test_df['project'].nunique()})")

    return test_df

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
    if not raw_input:
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

def handle_zero_n_few_shot_prompts_format(cigrate_config_path, input_yaml, migration_type, mode):
    # Read prompts from config folder
    base_config = os.path.join(cigrate_config_path, "prompts")
    def read_prompt(fname):
        fpath = os.path.join(base_config, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            return f.read()

    few_shot_mode = mode == "few_shot"
    # Configure conversion prompts based on target format
    # Each conversion type has specific system and user prompts for optimal results
    if migration_type == "travis_to_gha":
        system_prompt = read_prompt("travis_to_gha_system.txt")
        user_prompt_template = read_prompt("travis_to_gha_user.txt")
        example_block = ""
        if few_shot_mode:
            # Use the entire example file content as the block
            example_file_path = os.path.join(cigrate_config_path, "few-shot-examples/travis_to_gha.txt")
            if os.path.exists(example_file_path):
                with open(example_file_path, "r", encoding="utf-8") as ef:
                    example_block = ef.read()
        user_prompt = user_prompt_template.replace("{{ FEW_SHOT_EXAMPLES }}", example_block).replace("{{ INPUT_CODE }}", input_yaml)
    elif migration_type == "gha_to_travis":
        system_prompt = read_prompt("gha_to_travis_system.txt")
        user_prompt_template = read_prompt("gha_to_travis_user.txt")
        example_block = ""
        if few_shot_mode:
            example_file_path = os.path.join(cigrate_config_path, "few-shot-examples/gha_to_travis.txt")
            if os.path.exists(example_file_path):
                with open(example_file_path, "r", encoding="utf-8") as ef:
                    example_block = ef.read()
        user_prompt = user_prompt_template.replace("{{ FEW_SHOT_EXAMPLES }}", example_block).replace("{{ INPUT_CODE }}", input_yaml)
    else:
        raise ValueError("❌ Invalid conversion type. Use 'travis_to_gha' or 'gha_to_travis'.")

    return system_prompt, user_prompt

def generate_migration_openai(model_name, migration_type, mode, cigrate_config_path, input_yaml):
    instruction_text, input_text = handle_zero_n_few_shot_prompts_format(cigrate_config_path, input_yaml, migration_type, mode)

    prompt = [
        {"role": "system", "content": instruction_text},
        {"role": "user", "content": input_text}
    ]
    
    try:
        # Route to appropriate LLM service based on model name
        model_response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
            model=model_name,
            temperature=0,
            messages=prompt
        )
        raw_output = model_response.choices[0].message.content or ""
        generated_yaml = re.sub(r"^```(yaml|yml)?\s*", "", raw_output, flags=re.IGNORECASE).strip()
        generated_yaml = re.sub(r"\s*```$", "", generated_yaml)
        return generated_yaml.strip()
    except Exception as e:
        print(f"Error generating migration with OpenAI: {e}")
        return ""

def write_project_outputs(project_name, outputs, results_dir):
    project_dir = f"{results_dir}/{project_name}"
    os.makedirs(project_dir, exist_ok=True)
    # Ensure subdirs exist
    subdirs = [
        "01_Original_Travis",
        "02_Original_GHA",
        f"03_CIgrate_Travis_to_GHA/{model_name}",
        f"04_CIgrate_GHA_to_Travis/{model_name}"
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
        with open(f"{project_dir}/03_CIgrate_Travis_to_GHA/{model_name}/actions.yml", 'w') as f:
            f.write(outputs['travis_to_gha_generated'])
    if outputs.get('gha_to_travis_generated'):
        with open(f"{project_dir}/04_CIgrate_GHA_to_Travis/{model_name}/travis.yml", 'w') as f:
            f.write(outputs['gha_to_travis_generated'])

def evaluate_model(model_name, migration_type, mode, test_df, results_dir, cigrate_config_path):
    # Model evaluation with bidirectional testing + results saving

    # Determine which projects to evaluate
    all_projects = list(test_df['project'].unique())
    print(f"📊 Evaluating all {len(all_projects)} projects")

    # Build an index list of test_df rows belonging to the selected projects
    selected_indices = test_df.index[test_df['project'].isin(all_projects)].tolist()
    print(f"📄 Selected test rows: {len(selected_indices)}")

    print(f"📊 Testing {model_name} model on {len(selected_indices)} samples across {len(all_projects)} projects...")

    project_outputs = {}
    for j, i in enumerate(tqdm(selected_indices, desc=f"Generating a {migration_type.replace('_to_', '-to-')} migration")):
        # Get test data
        test_row = test_df.loc[i]
        project_name = test_row['project']
        direction = test_row['direction']
        instruction_text = test_row["instruction"]
        input_yaml = test_row["input"]
        ground_truth = test_row['output']

        subdir = "03_CIgrate_Travis_to_GHA" if direction == "travis_to_gha" else "04_CIgrate_GHA_to_Travis"
        out_name = "actions.yml" if direction == "travis_to_gha" else "travis.yml"
        output_yml_path = f"{results_dir}/{project_name}/{subdir}/{model_name}/{out_name}"

        # Skip if the output YAML file already exists
        if os.path.exists(output_yml_path):
            continue

        # Extract clean source and ground truth YAML
        source_yaml = extract_clean_source_yaml(input_yaml)
        ground_truth_yaml = extract_clean_source_yaml(ground_truth)

        # Generate migration and clean
        clean_generated_yaml = generate_migration_openai(model_name, migration_type, mode, cigrate_config_path, source_yaml)

        time.sleep(10)

        # Update project_outputs
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

        # Results save: write this project's current state after each sample
        write_project_outputs(project_name, project_outputs[project_name], results_dir)

    print(f"✅ Files written under: {results_dir}/")
    print(f"✅ Generated migrations for {len(project_outputs)} unique projects")

def run_cigrate_steps(model_name, migration_type, mode, test_path, results_dir, cigrate_config_path):
    test_df = timed_step("Data Preparation", "📚", 1, prepare_data, test_path)
            
    timed_step("Model Evaluation", "🧪", 2, evaluate_model, model_name, migration_type, mode, test_df, results_dir, cigrate_config_path)
    
if __name__ == "__main__":
    model_name = 'gpt-4o'
    cigrate_config_path = '/Users/taherghaleb/Downloads/CIgrate_Config'
    input_data_path = '/Users/taherghaleb/Downloads/CIgrate_FineTuning_Data/OurCleanData/Sample_153/Set_Full'
    output_data_path = '/Users/taherghaleb/Downloads/CIgrate_New_Results-OurCleanData-GPT-4o'
    migration_types = ['travis_to_gha', 'gha_to_travis']
    modes = ['zero_shot', 'few_shot']

    #model_name = sys.argv[2]
    #cigrate_config_path = sys.argv[6]
    #input_data_path = sys.argv[7]
    #output_data_path = sys.argv[8]
    #migration_type = sys.argv[9]
    #mode = sys.argv[10]

    # Start overall timing
    total_start_time = time.time()
    
    for migration_type in migration_types:
        for mode in modes:
            log_file = f'slurm_out/{model_name}_{migration_type}_{mode}_log.out'
            sys.stdout = Logger(log_file)

            print(f"🔄 Evaluating GPT-4o model for {migration_type} and {mode}...")

            folder = "CIgrate_Results_ZeroShot" if mode == "zero_shot" else "CIgrate_Results_FewShot"
            results_dir = f"{output_data_path}/{folder}/Sample_153/Set_Full"

            os.makedirs(results_dir, exist_ok=True)

            test_path  = os.path.join(input_data_path, f"instruction_dataset_{migration_type}.csv") # 'instruction_dataset_travis_to_gha.csv' 'instruction_dataset_gha_to_travis.csv'

            run_cigrate_steps(model_name, migration_type, mode, test_path, results_dir, cigrate_config_path)

            # Final timing summary for each migration type and mode
            total_time = time.time() - total_start_time
            print("=" * 87)
            print(f"🎉 All done for {migration_type} and {mode}!")
            print(f"⏱️  Total execution time for {migration_type} and {mode}: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
            print("=" * 87)