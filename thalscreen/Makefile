.PHONY: data benchmark test clean

data:
	python -c "from kagglehub import dataset_download" 2>/dev/null || echo "note: kagglehub not installed"
	python -c "\
import sys;\
sys.path.insert(0, 'src');\
from thalscreen.dataset import load_dataset;\
df = load_dataset('data/hplc_thal.zip');\
print('data ok:', df.shape)"

benchmark:
	python run_benchmark.py

benchmark-fast:
	python run_benchmark.py --skip-curve

test:
	python -m pytest tests/ -q

clean:
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache
