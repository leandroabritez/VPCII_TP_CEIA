import json
path = 'notebooks/04_evaluation.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for cell in data['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            line = line.replace("'yolov11_fracture'", "'exp1_yolo11_raw_colab'")
            # fix for rtdetr mapping
            line = line.replace("'rtdetr_fracture'", "'resultados_rtdetr/rtdetr_fracture'") 
            new_source.append(line)
        cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2)
