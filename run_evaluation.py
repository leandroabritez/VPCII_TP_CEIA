import json
with open('notebooks/04_evaluation.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

script = ""
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        # Comment out magic commands or display()
        source = source.replace("display(summary)", "print(summary)")
        script += source + "\n\n"

with open('04_evaluation_script.py', 'w', encoding='utf-8') as f:
    f.write(script)
