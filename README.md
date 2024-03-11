# SignGraph: A Sign Sequence is Worth Graphs of Nodes
An implementation of the paper: SignGraph: A Sign Sequence is Worth Graphs of Nodes. (CVPR 2024) [[paper]]()



## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- You can install other required modules by conducting 
   `pip install -r requirements.txt`

## Implementation
The implementation for the SignGraoh (line 18) is given in [./modules/resnet.py](https://github.com/hulianyuyy/CorrNet_CSLR/blob/main/modules/resnet.py).  

 
 

## Data Preparation
 
1. PHOENIX2014 dataset: Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). 

2. PHOENIX2014-T datasetDownload the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

3. CSL dataset： Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

 
Download datasets and extract them, no further data preprocessing needed. 

## Weights  

We make some imporvments of our code, and provide newest checkpoionts and better performance.

|Dataset | Backbone | Dev WER | Del / Ins | Test WER  | Del / Ins | Pretrained model                                            |
| --------| -------- | ---------- | ----------- | ----------- | -----------| --- |
|Phoenix14T | SignGraph |  17.00|04.99/ 02.32| 19.44| 05.14/03.38|[[Google Drive]](https://drive.google.com/drive/folders/1FVvbXV7f2-5lJhVlCm-bqzyZ55C1LQ-g?usp=sharing) |
|Phoenix14 |SignGraph|17.13|06.00/02.17| 18.17| 05.65/02.23|[[Google Drive]]() |
|CSL-Daily |SignGraph|-|-/-| -| -/-|[[Google Drive]]() |


​To evaluate the pretrained model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：   
`python main.py --device your_device --load-weights path_to_weight.pt --phase test`

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

`python main.py --device your_device`

Note that you can choose the target dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml.
 

This repo is based on [VAC (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html), [VIT (NIPS 2022)]() and [RTG-Net (ACM MM2023)]( ), Many thanks for their great work!

### Citation

If you find this repo useful in your research works, please consider citing:

```latex
@inproceedings{SignGraph,
  title={SignGraph: A Sign Sequence is Worth Graphs of Nodes},
  author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Xia, Kang and Xie, Lei and Lu, Sanglu},
  booktitle={CVPR2024},
  pages={763--772},
  year={2023}
}

@inproceedings{gan2023towards,
  title={Towards Real-Time Sign Language Recognition and Translation on Edge Devices},
  author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Xie, Lei and Lu, Sanglu},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={4502--4512},
  year={2023}
}


@inproceedings{gan2023contrastive,
  title={Contrastive learning for sign language recognition and translation},
  author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Xia, Kang and Xie, Lei and Lu, Sanglu},
  booktitle={Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, IJCAI-23},
  pages={763--772},
  year={2023}
}


 
```