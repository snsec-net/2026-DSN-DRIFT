from pathlib import Path

path_dir_root = Path(__file__).parent.parent

path_data = path_dir_root.joinpath('dataset')
path_data.mkdir(exist_ok=True, parents=True)

path_period_data = path_data.joinpath('DRIFT_input_eSLD')
path_dga_scheme = path_data.joinpath('dga_scheme')
assert path_period_data.exists(), 'DRIFT_input_eSLD does not exist'

path_tokenizer = path_dir_root.joinpath('artifacts/tokenizer')
path_tokenizer.mkdir(exist_ok=True, parents=True)
path_model = path_dir_root.joinpath('artifacts/model')
path_model.mkdir(exist_ok=True, parents=True)

path_figure = path_dir_root.joinpath(f'figure')
path_figure.mkdir(exist_ok=True, parents=True)