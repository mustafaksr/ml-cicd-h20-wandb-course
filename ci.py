import accelerate
print(f'The version of wandb is: {accelerate.__version__}')
assert accelerate.__version__ == '0.25.5', f'Expected version 0.25.5, but got {accelerate.__version__}'

