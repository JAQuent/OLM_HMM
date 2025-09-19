from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()

PATHS = {
        'raw_data': PROJECT_DIR / 'data' / 'raw',
        'processed_data': PROJECT_DIR / 'data' / 'processed', 
        'figures': PROJECT_DIR / 'results' / 'figures',
        'logs': PROJECT_DIR / 'results' / 'logs',
        'src_python': PROJECT_DIR / 'src' / 'python',
        'src_r': PROJECT_DIR / 'src' / 'r'       
        }

# Ensure directories exist
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)
