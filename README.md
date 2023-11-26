## Masked Autoencoders for saliency prediction

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/146857310-f258c86c-fde6-48e8-9cee-badd2b21bd2c.png" width="480">
</p>


```
@Article{MaskedAutoencoders2021,
  author  = {Kaiming He and Xinlei Chen and Saining Xie and Yanghao Li and Piotr Doll{\'a}r and Ross Girshick},
  journal = {arXiv:2111.06377},
  title   = {Masked Autoencoders Are Scalable Vision Learners},
  year    = {2021},
}
```




We used pretrained MAE encoder to train the decoder for the task of masked image saliency prediction.
If you want to run the pretraining yourself, you need to download the SALICON data (http://salicon.net/challenge-2017/)
```
data
└── salicon
    ├── fixations-2
    │   ├── train
    │   ├── val
    │   └── test
    ├── images
    │   ├── train
    │   ├── val
    │   └── test
    └── maps-2
        ├── train
        └── val
        
```

### Visualization demo

Run our interactive visualization demo (/demo/mae_visualize_saliency.ipynb) using [Colab notebook](https://colab.research.google.com/github/facebookresearch/mae/blob/main/demo/mae_visualize.ipynb) (no GPU needed):


