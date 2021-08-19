### Requirements:

- PyTorch >=1.4
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- [detectron2](https://github.com/facebookresearch/detectron2)

Our model are mainly based on detectron2 architecture. The main modifications are at [meta_arch](https://github.com/ispc-lab/KAM-Net/tree/master/detectron2/modeling/meta_arch) and [backbone](https://github.com/ispc-lab/KAM-Net/tree/master/detectron2/modeling/backbone). User can replace the original part with ours.

### Training
```bash
python train.py
```

### Test
```bash
python eval_my.py
```
