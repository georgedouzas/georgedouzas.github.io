set positional-arguments

# List commands
@default:
  just --list

# Clean cache of project
@clean category project:
    rm -rf .quarto/_freeze/projects/$1/$2/notebooks/.jupyter_cache
    rm -rf _freeze/projects/$1/$2/notebooks/.jupyter_cache
    rm -rf projects/$1/$2/notebooks/.jupyter_cache

# Render project
@render category project:
    source .venv/bin/activate
    pip install -r projects/$1/$2/src/requirements.txt 
    quarto render projects/$1/$2/notebooks
