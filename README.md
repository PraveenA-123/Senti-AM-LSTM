# Senti-AM-LSTM

SENTI-AM-LSTM: A novel sentiment-enhanced attention-based LSTM model for agricultural futures price prediction.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/senti-am-lstm-tool.git
   cd senti-am-lstm-tool


2. Install the package locally:
   ```bash
   pip install -e .
   ```

## Usage

Once installed, you can use the `senti-am-lstm` command to predict prices:

```bash
senti-am-lstm --input_file <path_to_input_file.xlsx> --output_file <path_to_output_file.xlsx>
```

Example:

```bash
senti-am-lstm --input_file C:/path/to/input.xlsx --output_file C:/path/to/output.xlsx
```

## Project Structure

```
Senti_AM_LSTM_tool/
├── senti_am_lstm_tool/
│   ├── __init__.py
│   ├── predict_price.py
│   └── model_utils.py
├── models/
│   ├── attention_model.h5
│   ├── scaler_features.pkl
│   └── scaler_target.pkl
├── setup.py
└── README.md
```
```

### Final Notes:

- After setting up the project, run `pip install -e .` to install the package locally.
- The command-line tool can then be run using `senti-am-lstm`.
- The models and scalers must be placed inside the `models/` folder.

This setup ensures that the tool loads models automatically from the `models/` folder, and users only need to specify the input and output files when running the script from the command line. 
