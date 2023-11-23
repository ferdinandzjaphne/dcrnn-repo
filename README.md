Dependency can be installed using the following command:
```bash
pip install -r requirements.txt
```

Run the following commands to start training
```bash
# urban-core
python -m train_model --config_filename=config.yaml --module=urban_core

# urban-mix
python -m train_model --config_filename=config.yaml --module=urban_mix
```

