# 1️⃣ Deactivate any existing environments
deactivate 2>/dev/null || true

# 2️⃣ Fix PATH for local scripts
export PATH="$HOME/.local/bin:$PATH"

# 3️⃣ Load modules
module load python/3.13.2
module load gcc arrow
module load cuda

# 4️⃣ Remove old environment
rm -rf ~/envs/unsloth_env

# 5️⃣ Create virtual environment
python3 -m venv ~/envs/unsloth_env

# 6️⃣ Activate environment
source ~/envs/unsloth_env/bin/activate

# 7️⃣ Upgrade pip, wheel, setuptools
pip install --upgrade pip wheel setuptools

# 8️⃣ Install compatible Python packages (PyTorch already comes from module)
pip install torch>=2.8.0 triton>=3.4.0 torchvision bitsandbytes
pip install transformers==4.56.2 tokenizers trl==0.22.2
pip install --upgrade --force-reinstall --no-deps --no-cache-dir "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo.git"
pip install --upgrade --force-reinstall --no-deps --no-cache-dir "unsloth[base] @ git+https://github.com/unslothai/unsloth.git"
pip install git+https://github.com/triton-lang/triton.git@05b2c186c1b6c9a08375389d5efe9cb4c401c075#subdirectory=python/triton_kernels

pip install numpy pandas datasets typing bitsandbytes trl numpy matplotlib seaborn scikit-learn scipy statsmodels nltk huggingface-hub peft protobuf sentencepiece cut_cross_entropy hf_transfer msgspec packaging pillow psutil torchao tyro

deactivate 2>/dev/null || true


