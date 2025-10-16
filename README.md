# SignGraph: A Sign Sequence is Worth Graphs of Nodes
An implementation of the paper: SignGraph: A Sign Sequence is Worth Graphs of Nodes. (CVPR 2024) [[paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Gan_SignGraph_A_Sign_Sequence_is_Worth_Graphs_of_Nodes_CVPR_2024_paper.pdf)

# MixSignGraph: MixSignGraph: A Sign Sequence is Worth Mixed Graphs of Nodes

A new CSLR, SLT, Gloss-free SLT model are coming

## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus please install Pytorch first.

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.

- For these who failed install ctcdecode (and it always does), you can download [ctcdecode here](https://drive.google.com/file/d/1LjbJz60GzT4qK6WW59SIB1Zi6Sy84wOS/view?usp=sharing), unzip it, and try    
`cd ctcdecode` and `pip install .`

- Please follow [this link](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install pytorch geometric

- You can install other required modules by conducting 
   `pip install -r requirements.txt`

 

 

## Data Preparation
 
1. PHOENIX2014 dataset: Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/). 

2. PHOENIX2014-T datasetDownload the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)

3. CSL dataset： Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

 
Download datasets and extract them, no further data preprocessing needed. 

## Weights  

We make some imporvments of our code, and provide newest checkpoionts and better performance.

|Dataset | Backbone | Dev WER | Del / Ins | Test WER  | Del / Ins | Pretrained model                                            |
| --------| -------- | ---------- | ----------- | ----------- | -----------| --- |
|Phoenix14T | SignGraph |  17.00|4.99/2.32| 19.44| 5.14/3.38|[[Google Drive]](https://drive.google.com/drive/folders/1FVvbXV7f2-5lJhVlCm-bqzyZ55C1LQ-g?usp=sharing) |
|Phoenix14 |SignGraph|17.13|6.00/2.17| 18.17|5.65/2.23|[[Google Drive]](https://drive.google.com/drive/folders/1O5JBkmnu2TO8Domzd60tqql8l1zNCzHc?usp=sharing) |
|CSL-Daily |SignGraph|26.38|9.92/2.62| 25.84|9.39/2.58|[[Google Drive]](https://drive.google.com/drive/folders/1t09Ixpiujw6WJrkSF8gwexvKJie4RGsh?usp=sharing) |



​To evaluate the pretrained model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：   
`python main.py --device your_device --load-weights path_to_weight.pt --phase test`

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:

`python main.py --device your_device`

Note that you can choose the target dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml.
 
### Thanks

This repo is based on [VAC (ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Min_Visual_Alignment_Constraint_for_Continuous_Sign_Language_Recognition_ICCV_2021_paper.html), [VIT (NIPS 2022)](https://arxiv.org/abs/2206.00272) and [RTG-Net (ACM MM2023)](https://dl.acm.org/doi/10.1145/3581783.3611820)！

### Citation

If you find this repo useful in your research works, please consider citing:

```latex
@inproceedings{gan2024signgraph,
  title={SignGraph: A Sign Sequence is Worth Graphs of Nodes},
  author={Gan, Shiwei and Yin, Yafeng and Jiang, Zhiwei and Wen, Hongkai and Xie, Lei and Lu, Sanglu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13470--13479},
  year={2024}
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

@article{han2022vision,
  title={Vision gnn: An image is worth graph of nodes},
  author={Han, Kai and Wang, Yunhe and Guo, Jianyuan and Tang, Yehui and Wu, Enhua},
  journal={Advances in neural information processing systems},
  volume={35},
  pages={8291--8303},
  year={2022}
} 
```
