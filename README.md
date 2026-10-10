# hackathon

## Run the synthetic next-step training pipeline

Create and activate the virtual environment, install the project dependencies,
and run the training script:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

The script generates 256 reproducible sequences. Each `X` sample has shape
`(10, 19)` and its `y` target is the next 19-value vector. It splits the samples
into training, validation, and test sets, batches them, trains both reservoirs,
and reports MSE and MAE.