"""
download_dataset.py – Descarga el dataset Bone Fracture Detection desde Roboflow.

Uso:
    python scripts/download_dataset.py --api-key TU_API_KEY
    
    # O con variable de entorno:
    $env:ROBOFLOW_API_KEY = "TU_API_KEY"
    python scripts/download_dataset.py
"""

import argparse
import os
import sys
from pathlib import Path

# Cargar .env automáticamente si existe
try:
    from dotenv import load_dotenv
    env_file = Path(__file__).parent.parent / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        print(f'✓ .env cargado desde {env_file}')
except ImportError:
    pass  # python-dotenv no instalado, usar env vars del sistema

def main():
    parser = argparse.ArgumentParser(description='Descargar dataset de fracturas desde Roboflow.')
    parser.add_argument('--api-key', default=os.getenv('ROBOFLOW_API_KEY', ''),
                        help='API key de Roboflow (o setear ROBOFLOW_API_KEY)')
    parser.add_argument('--version', type=int, default=4,
                        help='Versión del dataset (default: 4)')
    parser.add_argument('--format', default='yolov11',
                        help='Formato de descarga (default: yolov11)')
    args = parser.parse_args()

    if not args.api_key:
        print('❌ API key no encontrada.')
        print('   Opciones:')
        print('   1. python scripts/download_dataset.py --api-key TU_KEY')
        print('   2. $env:ROBOFLOW_API_KEY = "TU_KEY"; python scripts/download_dataset.py')
        print('\n   Obtener key gratis en: https://app.roboflow.com → Settings → API Keys')
        sys.exit(1)

    from roboflow import Roboflow

    OUTPUT_DIR = Path(__file__).parent.parent / 'data'
    OUTPUT_DIR.mkdir(exist_ok=True)

    print('🔗 Conectando a Roboflow...')
    rf = Roboflow(api_key=args.api_key)
    project = rf.workspace("veda").project("bone-fracture-detection-daoon")

    print(f'📦 Descargando versión {args.version} en formato {args.format}...')
    dataset = project.version(args.version).download(
        model_format=args.format,
        location=str(OUTPUT_DIR / 'bone-fracture-detection-daoon-1'),
        overwrite=True,
    )

    print(f'\n✅ Dataset descargado exitosamente en:')
    print(f'   {OUTPUT_DIR / "bone-fracture-detection-daoon-1"}')
    print('\nEstructura esperada:')
    for split in ['train', 'valid', 'test']:
        split_path = OUTPUT_DIR / 'bone-fracture-detection-daoon-1' / split
        if split_path.exists():
            imgs = len(list((split_path / 'images').glob('*'))) if (split_path / 'images').exists() else 0
            print(f'   {split:>6}/: {imgs} imágenes')

if __name__ == '__main__':
    main()
