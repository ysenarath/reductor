rsync -rtuv --delete \
    --exclude='.ipynb_checkpoints' \
    --exclude='outputs' \
    --exclude='mlruns' \
    --exclude='notebooks' \
    --exclude='logs' \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude ".DS_Store" \
    --exclude ".git" \
    --exclude ".mypy_cache" \
    --exclude ".vscode" \
    --exclude ".ruff_cache" \
    --exclude ".gradio" \
    --exclude "orc" \
    ./ ywijesu@hopper.orc.gmu.edu:/scratch/ywijesu/projects/other/reductor/