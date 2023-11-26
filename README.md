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

Run the following commands to show training results
```bash
# urban-core
python show_result.py --config_filename=data/model/pretrained/urban-core/config.yaml --module=urban_core

# urban-mix

```