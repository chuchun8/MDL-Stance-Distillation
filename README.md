# MDL-Stance-Distillation

EMNLP 2021 main conference paper: Improving Stance Detection with Multi-Dataset Learning and Knowledge Distillation.

## Abstract

Stance detection determines whether the author of a text is in favor of, against or neutral to a specific target and provides valuable insights into important events such as legalization of abortion. Despite significant progress on this task, one of the remaining challenges is the scarcity of annotations. Besides, most previous works focused on a hard-label training in which meaningful similarities among categories are discarded during training. To address these challenges, first, we evaluate a multi-target and a multi-dataset training settings by training one model on each dataset and datasets of different domains, respectively. We show that models can learn more universal representations with respect to targets in these settings. Second, we investigate the knowledge distillation in stance detection and observe that transferring knowledge from a teacher model to a student model can be beneficial in our proposed training settings. Moreover, we propose an Adaptive Knowledge Distillation (AKD) method that applies instance-specific temperature scaling to the teacher and student predictions. Results show that the multi-dataset model performs best on all datasets and it can be further improved by the proposed AKD, outperforming the state-of-the-art by a large margin.

## Dataset

SemEval-2016, AM, Multi-Target, WT-WT and Covid datasets are used for evaluation. Please refer the original paper to the train/val/test splits.

## Run

BERTweet is used as our base model in this paper. First, configure the environment:
```
$ pip install -r requirements.txt
```
For teacher model trained on all datasets, run
```
cd src/
python train_model.py \
    --input_target all \
    --model_select Bertweet \
    --train_mode unified \
    --col Stance1 \
    --model_name teacher \
    --dataset_name all \
    --filename Stance_All_Five_Datasets \
    --lr 2e-5 \
    --batch_size 32 \
    --epochs 5 \
```
For AKD student model trained on all datasets, then run
```
cd src/
python train_model.py \
    --input_target all \
    --model_select Bertweet \
    --train_mode unified \
    --col Stance1 \
    --model_name student \
    --dataset_name all \
    --filename Stance_All_Five_Datasets \
    --lr 2e-5 \
    --batch_size 32 \
    --epochs 5 \
    --dropout 0. \
    --alpha 0.7 \
    --theta 0.6 \
```
`model_select` includes two options: [`Bertweet` and `Bert`].

`train_mode` includes two options: [`unified` and `adhoc`]. It is set to be multi-target training setting when `train_mode` is `unified` and `dataset_name` is single dataset. It is multi-dataset training when `train_mode` is `unified` and `dataset_name` is `all`.

`col` indicates the target in each target-pair and it should be `Stance1` in most cases. We have `Stance2` only when we evaluate the methods on Multi-Target dataset.

`alpha` balances the importance of cross-entropy loss and KL divergence loss in knowledge distillation.

`theta` is the number of `a1` mentioned in this paper.


## Contact Info

Please contact Yingjie Li at yli300@uic.edu with any questions.
