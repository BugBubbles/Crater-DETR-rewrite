# Description

This repository is a REWRITE version of "Crater-DETR: a novel transformer network for crater detection based on dense supervision and multiscale fusion", not an official version. Up to now I cannot get same result as what the authors did. Maybe the dataset and networks detail matters. So welcome for any discussion and pull request!

# Result

This result below attain from MDCD dataset, refer to "High-resolution feature pyramid network for automatic crater detection on Mars". You can download it from their official [site](https://doi.org/10.5281/zenodo.4750929).

| epoch | mAP   |
| ----- | ----- |
| 50    | 0.321 |

## Installation

First install the `lastest` mmedetection following: [https://mmdetection.readthedocs.io/en/latest/get_started.html](https://mmdetection.readthedocs.io/en/latest/get_started.html)

Then go to `mmdetection/projects`, clone this repostry:

```bash
git clone https://github.com/BugBubbles/Crater-DETR-rewrite ./Crater-DETR
```

Then you can use this repostry as any one in `projects`. Maybe you should first deteminate your datasets links in `./datasets/*.py` files. Or you can download the Dataset from ``.

# Reference

```bibtex
@article{guoCraterDETRNovelTransformer2024,
  title = {Crater-{{DETR}}: A Novel Transformer Network for Crater Detection Based on Dense Supervision and Multiscale Fusion},
  shorttitle = {Crater-{{DETR}}},
  author = {Guo, Yue and Wu, Hao and Yang, Shuojin and Cai, Zhanchuan},
  date = {2024-03-11},
  journaltitle = {IEEE Transactions on Geoscience and Remote Sensing},
  shortjournal = {IEEE Trans. Geosci. Remote Sensing},
  volume = {62},
  pages = {1--12},
  issn = {0196-2892, 1558-0644},
  doi = {10.1109/TGRS.2024.3376398},
  langid = {english},
}
```
