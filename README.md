# uGRU - unified Gated-Recurrent-Unit for automated gait event detection.

- Train an automated gait event detection algorithm using your own labelled dataset.
- Run prediction on test dataset using a trained model.
- Perform gait analysis (spatio-temporal features) using detected gait events.

## 1. Prerequisites:
```    
git clone <repo>
# create virtual environment
uv venv
# install package
python -m pip install -e .
``` 

## 2. Quickstart
- From the project root, use the Hydra-backed scripts in `scripts/` and `src/gait_ml/`.

### 2.1 (TBD) How to train 
- Run training (Hydra overrides supported):

    <!-- ```bash
    python scripts/train.py \
    general.data_path='data/**/*.npy' \
    general.group_file='data/groups.csv' \
    general.project_name='myproject' \
    data.batch_size=64 \
    training.num_epochs=50
    ``` -->

### 2.2 How to run predictions
<!-- - Using the package module (recommended when package is on PYTHONPATH):
    ```bash
    python predict_gait_events.py ckpt_path=/abs/path/model.ckpt files_glob="../data/**/**.xls"
    # optional overrides
    python predict_gait_events.py device=cuda batch_size=64 window_size=256 step_size=256
    ``` -->

- Programmatic inference:
    ```python
    from gait_ml.predict import predict_gait_event_labels
    from glob import glob

    files = glob("../data/dataset2/10092025/imu/Mobilephone/Termin_1_Vicon/ID_01_T1/*_2mW_IPhone.xls")
    model_fpath = "/home/qivy00li/projects/gait_ml/backpain/ZscaledRerunExp4-Fold1-GRU-expandlabel2_2025-11-10_15-13-18/checkpoints/model-epoch=97-val_f1score=0.88.ckpt"
    merged_prediction = predict_gait_event_labels(model_fpath, files)
    ```

<!-- Notes
- Hydra override keys use the `predict.`, `data.` and `training.` namespaces from configs.
- Adjust `window_size`, `step_size`, and model/data params to match training settings. -->