# Installation

```bash
pip install (-e) .
```

# Running Tests

```bash
py.test .
```



## Alphapose Inference Callback

* runs alphapose model on provided image dir after eval op and saves alphapose results in eval folder

```python
import alphapose_callback
class Iterator(TemplateItertor):
    def __init__(...):
        ...

    @property
    def callbacks(self):
        return {"eval_op": {"pose_inference": alphapose_callback.inference_callback}}
```


```yaml
alphapose_callback: 
    alphapose_python: /abspath/anaconda3/envs/alphapose/bin/python
    infer_script: /abspath/AlphaPose/scripts/demo_inference.py
    config: /abspath/AlphaPose/configs/coco/hrnet/256x192_w32_lr1e-3.yaml
    checkpoint: /abspath/AlphaPose/pretrained_models/hrnet_w32_256x192.pth
    alphapose_dir: /abspath/AlphaPose/
    indir: transfer_batch
    outdir: alphapose
```


## Alphapose PCK Callback

* runs alphapose model on provided image dir after eval op and saves alphapose results in eval folder
* then loads alphapose results and compares against

```python
import alphapose_callback
class Iterator(TemplateItertor):
    def __init__(...):
        ...

    @property
    def callbacks(self):
        return {"eval_op": {"pck": alphapose_callback.pck_callback}}
```

```yaml
alphapose_callback: 
    alphapose_python: /abspath/anaconda3/envs/alphapose/bin/python
    infer_script: /abspath/AlphaPose/scripts/demo_inference.py
    config: /abspath/AlphaPose/configs/coco/hrnet/256x192_w32_lr1e-3.yaml
    checkpoint: /abspath/AlphaPose/pretrained_models/hrnet_w32_256x192.pth
    alphapose_dir: /abspath/AlphaPose/
    indir: transfer_batch
    outdir: alphapose
alphapose_pck_callback: 
    true_poses_file: /abspath/alphapose_true_poses.json
    distance_threshold: [10, 20, 30] # will run calculation with all these parameters
```


# Callback in Action

```bash

```

![assets/callback_in_action.png](assets/callback_in_action.png)