# Sample Run Flow

## Run the full pipeline

```bash
python main.py run-full-pipeline --config config/dev.yaml
```

## Inspect the selected alphas

```bash
python main.py top --config config/dev.yaml --limit 10
python main.py report --config config/dev.yaml --limit 5
```

## Generate additional mutations from the current top set

```bash
python main.py mutate --config config/dev.yaml --from-top 10 --count 50
python main.py evaluate --config config/dev.yaml
python main.py top --config config/dev.yaml --limit 10
```

## Step-by-step workflow

```bash
python main.py load-data --config config/dev.yaml
python main.py generate --config config/dev.yaml --count 60
python main.py evaluate --config config/dev.yaml
python main.py top --config config/dev.yaml --limit 10
```
