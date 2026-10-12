# VIC Stock Price Forecasting with LTSF-Linear Models

An educational PyTorch notebook for comparing three linear time-series forecasting models on historical VIC stock data. The task uses 7, 30, 120, or 480 historical observations to predict the next 7 closing-price observations.

**Project status:** this is a hint notebook, not a completed implementation. The model-definition cell contains 16 `***Your Code Here***` placeholders. These are intentionally preserved and must be replaced before training. The translated notebook retains the original saved outputs; they are historical examples, not results from a new execution of the translated file.

## Files

| File | Purpose |
| --- | --- |
| `VIC_LTSF_Linear_Forecasting.ipynb` | English notebook containing data exploration, model exercises, training, evaluation, and plots |
| `README.md` | Project description, environment setup, execution instructions, and implementation notes |
| `[Data]-VIC.csv` | Required input data, downloaded by the notebook or supplied locally; not included in this delivery |

## Purpose and scope

The project demonstrates how simple linear models can forecast a univariate time series, how input-window length affects performance, and how decomposition and last-value normalization change model behavior. LTSF stands for long-term time-series forecasting, although this exercise uses a fixed seven-step prediction horizon.

The notebook calls observations “days.” In the code, a day means one consecutive data row, not a calendar interval. With daily trading data, seven steps correspond to seven recorded trading observations.

The introductory objective mentions baseline comparisons. The implemented comparison framework contains Linear, DLinear, and NLinear only; separate persistence, moving-average, or other baseline models are not included. The notebook evaluates historical test windows and does not contain a separate routine for forecasting beyond the final dataset row.

## Dataset

The notebook downloads a CSV using this Google Drive file ID:

```text
1ZRfV1uMSDYokV_Mecv46h76MhbfHTHL8
```

It expects the downloaded file to be named `[Data]-VIC.csv` in the notebook's working directory. The original saved output reports 1,500 rows and seven columns, spanning August 3, 2020 through August 1, 2025. These details describe the saved example; the external download was not verified during translation.

| Column | Meaning and use |
| --- | --- |
| `time` | Observation date; parsed as a datetime and sorted chronologically |
| `open`, `high`, `low` | Price fields retained for inspection |
| `close` | Closing price; the source of the forecast target |
| `volume` | Trading volume; used in exploratory plots |
| `symbol` | Stock identifier; the example contains VIC |

The pipeline requires `time`, `close`, and `volume` for its current operations. Closing prices must be positive for the logarithm, and dates must be parseable. The original plot labels price as VND, but the notebook does not document the CSV's unit multiplier or adjustment methodology; verify those with the data provider before interpreting monetary values. Dataset provenance and licensing are not specified in the notebook.

## Environment setup

Use a Python 3 environment with JupyterLab or Google Colab. A CPU is sufficient to run the code once the exercise is completed; PyTorch automatically uses CUDA when available. The saved notebook metadata identifies a Colab T4 GPU, but a T4 is not required.

The original notebook does not pin Python or package versions. The commands below create a starting environment, not an environment reproduced from a tested lockfile.

### Local setup

Run in a terminal from the folder containing the notebook:

```bash
python -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

Or in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and launch JupyterLab:

```bash
python -m pip install --upgrade pip
python -m pip install jupyterlab ipykernel numpy pandas matplotlib seaborn scikit-learn torch gdown
python -m ipykernel install --user --name vic-ltsf --display-name "Python (VIC LTSF)"
python -m jupyter lab
```

Open `VIC_LTSF_Linear_Forecasting.ipynb` and select the **Python (VIC LTSF)** kernel. GPU execution requires a PyTorch installation compatible with the local CUDA environment; the notebook also supports CPU execution.

### Google Colab

1. Upload the notebook to Colab.
2. Optionally select a GPU runtime.
3. If dependencies are missing, run this in a new cell:

   ```python
   %pip install numpy pandas matplotlib seaborn scikit-learn torch gdown
   ```

4. Run the notebook's download cell, or upload `[Data]-VIC.csv` into the working directory and skip that cell.

The download requires internet access and permission to access the source file. If it fails, obtain the CSV from the project owner and provide it locally under the expected filename.

## Running the exercise

1. Run Sections 1–3 to download/load the data, inspect features, and create sliding-window datasets.
2. Complete every `***Your Code Here***` placeholder in Section 4. These placeholders are invalid Python syntax until replaced.
3. Run the model-definition and model-configuration cells.
4. Run Section 5 to normalize data and train all 12 model/window combinations.
5. Run Section 6 to view predictions and the performance summary.

After editing model definitions or configurations, restart the kernel and run cells in order to avoid stale model objects. Run All cannot complete while the exercise placeholders remain.

## Pipeline and model details

The notebook sorts observations by date, computes `daily_return = close.pct_change()` and `close_log = log(close)`, and fills missing daily returns with zero. Dataset construction then drops rows containing missing values. Only `close_log` is supplied to the forecasting models; returns, volume, and the other price columns are not model inputs.

Each sample contains an input vector of length `seq_len` and a target vector of length 7. For a cleaned series of length N, the number of windows is `N - seq_len - 7 + 1`. Model tensors have shape `[batch_size, seq_len]` at input and `[batch_size, 7]` at output.

| Model | Intended operation to implement |
| --- | --- |
| Linear | Apply one learned linear mapping from the historical window to the forecast horizon |
| DLinear | Extract a moving-average trend, subtract it to obtain the seasonal/residual component, forecast each component with a separate linear layer, and add the predictions |
| NLinear | Subtract the final input value, apply a linear projection, then add that value back to the forecast |

DLinear uses a default moving-average window of 5, replicate padding, and interpolation if needed to match the input length. NLinear's last-value normalization is separate from the dataset-level `StandardScaler`.

### Default configuration

| Setting | Value |
| --- | --- |
| Input lengths | 7, 30, 120, 480 observations |
| Forecast horizon | 7 observations |
| Split | 70% training, 15% validation, remaining 15% test, in window order |
| Target | Natural logarithm of closing price |
| Scaling | One `StandardScaler` reused across input lengths |
| Batch size | 32 |
| Training epochs | 50 per model/window combination |
| Loss | Mean squared error |
| Optimizer | Adam, learning rate 0.001 |
| Scheduler | StepLR, multiplying learning rate by 0.5 every 20 epochs |
| DataLoader behavior | Training shuffled; validation/test ordered; all use `drop_last=True` |

Input lengths appear in both `seq_lengths` and `horizon_configs`, as well as the training and plotting loops. Update them consistently when modifying experiments. Changing the forecast horizon also requires updating hard-coded `pred_len = 7` values, configurations, reshaping logic, and labels.

## Outputs and interpretation

The notebook produces exploratory price/return/volume plots, training and validation loss curves, day-seven prediction plots, average-seven-step prediction plots, and a table of RMSE, MAE, and R². MSE is also stored in `results`.

Evaluation reverses standardization but does not reverse the logarithm. Metrics therefore describe **log closing prices**, not raw currency prices. Applying `np.exp` to the inverse-standardized predictions converts them back to the CSV's price scale. The average prediction plot averages logarithms rather than raw prices.

Metrics flatten all seven forecast steps across all evaluated windows. Day-seven plots show only the final step of each window, so their displayed RMSE labels refer to the full seven-step evaluation. The x-axis is the prediction sample index, not a calendar date.

Trained models, metrics, and loss histories remain in memory in `trained_models`, `results`, and `loss_history`. No model checkpoints or CSV reports are automatically exported.

## Evaluation limitations

These behaviors are present in the supplied exercise and were preserved during translation:

- **Scaling leakage:** the scaler is fitted on the full first dataset, including validation and test windows, before splitting. A stricter experiment should fit scaling only on training observations.
- **Boundary overlap:** splits are chronological by window index, but adjacent windows share target dates across split boundaries. Date-based target boundaries or a suitable gap are needed for disjoint evaluation targets.
- **Different test periods:** each input length has a different number of windows, so percentage splits do not guarantee identical evaluation dates across input lengths.
- **Dropped samples:** `drop_last=True` discards incomplete validation/test batches. A split with fewer than 32 samples produces no batches. Use enough data for every split, or adjust batch handling when extending the exercise.
- **Reproducibility:** no random seed, early stopping, or best-validation checkpoint selection is implemented; evaluation uses the final epoch's model.

The saved outputs should be treated as examples of the reporting format, not independently verified performance claims.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `SyntaxError` in the model cell | Complete all 16 exercise placeholders |
| CSV not found | Confirm the filename and notebook working directory; run the download or upload the file manually |
| Missing package or `gdown` command | Install dependencies in the active notebook kernel/environment |
| `seaborn-v0_8` style unavailable | Use a Matplotlib version providing that style, or change the style to `default` |
| Missing values remain after filling returns | Some pandas versions handle chained inplace operations differently; if needed, use `df['daily_return'] = df['daily_return'].fillna(0)` |
| Empty loaders or metric errors | Check cleaned row counts, input length, split sizes, and `drop_last=True` |
| Non-finite log values | Check that closing prices are numeric, finite, and strictly positive |
| Results differ from saved examples | Check data, completed model code, package versions, randomness, and device |

## Translation and validation

Vietnamese headings, instructions, code comments, printed messages, and saved text outputs were translated into English. Existing English code identifiers, model placeholders, calculations, saved figures, and example results were retained. Notebook JSON structure and syntax of complete Python cells were checked. The download cell uses notebook shell syntax, and the model exercise intentionally remains incomplete. No fresh model training or end-to-end execution was performed.
