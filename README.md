# MEX: Memory-efficient Approach to Referring Multi-Object Tracking


## Requirements
```bash
conda create python=3.10 -n MEX --y
conda activate MEX
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia --y
conda install matplotlib pyyaml scipy tqdm tensorboard
pip install six==1.16.0
pip install transformers
pip install einops==0.7.0
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git 
pip install opencv-python
```
## Instalation for MeMOTR
```shell
# From https://github.com/fundamentalvision/Deformable-DETR
cd ./model/MeMOTR/models/ops

sh make.sh # for linux
python setup.py build install # for windows
# You can test this ops if you need:
python test.py
```

## Instalation for Trackeval
```shell
# From https://github.com/fundamentalvision/Deformable-DETR
cd ./TrackEval
pip install -r minimum_requirements.txt
pip install pycocotools scipy
```




## Prepare:

### Resources:
You can get all resources from: [MEX_Resource.zip](https://mega.nz/file/Im8nhAKa#qdT7bX42p30M9b3S-WqdOtUPNfb9gJkZY2omJx_hqds) with:
| Hash    | Value |
| -------- | ------- |
| MD5  | 5aec964670a9078bd908e8826b6ea631    |
| SHA256 | d482dd369003a4496a3a713f92e744d7bb0e3cf14716a46bf143fa7dfe3c8943    |


### Structures:
* Your root directory should be:
```
- checkpoints/
    CLIP/
        RN50.pt
        ViT-B-32.pt    
    IKUN/
        iKUN_cascade_attention.pth       
    MeMOTR/
        memotr_bdd100k.pth
        memotr_mot17.pth
    MEX/
        MEX_99.pth
...
- data_dir/
    Refer_Kitti/
        expression/
        KiTTI/
        Refer-KITTI_labels.json
        textual_features.json
...
- track-dataset/
    gt_template/
        0005/
        0011/
        0013/
    NeuralSORT/
        0005/
        0011/
        0013/
...
- main.py
```

## Training
```shell
python main.py --mode=train --module_name=MEX
```

## Submit
```shell
python main.py --mode=submit --module_name=MEX --module_checkpoint=./checkpoints/MEX/MEX_99.pth --track_root=./track-dataset --track_name=NeuralSORT
```

## Run inferences with MeMOTR

### MEX
```shell
python main.py --mode=inf --module_name=MEX --module_checkpoint=./checkpoints/MEX/MEX_99.pth --memotr_config=./configs/memotr_bdd100k.yaml --memotr_checkpoint=./checkpoints/MeMOTR/memotr_bdd100k.pth --video_path=<your_video> --caption=people-waking-on-the-street --module_threshold=0 --tracker_threshold=0.2
```

### IKUN
```shell
python main.py --mode=inf --module_name=IKUN --module_checkpoint=./checkpoints/IKUN/iKUN_cascade_attention.pth --memotr_config=./configs/memotr_bdd100k.yaml --memotr_checkpoint=./checkpoints/MeMOTR/memotr_bdd100k.pth --video_path=<mp4_video_path> --caption=people-waking-on-the-street --module_threshold=0 --tracker_threshold=0.2
```

## Run submit with MeMOTR
```shell
python main.py --mode=submitinf --module_name=MEX --module_checkpoint=./checkpoints/MEX/MEX_99.pth --memotr_config=./configs/memotr_bdd100k.yaml --memotr_checkpoint=./checkpoints/MeMOTR/memotr_bdd100k.pth --video_src=<your_folder_video_source> --module_threshold=0 --tracker_threshold=0.2 --inf_w_mem=True
```

## Run eval HOTA 

Noted to run  `submit` or `submitinf` command to generate submit folder, and feed to `<submit output folder>`

```shell
cd ./TrackEval/scripts

python run_mot_challenge.py \
--METRICS HOTA CLEAR Identity \
--SEQMAP_FILE ../seqmap.txt \
--SKIP_SPLIT_FOL True \
--GT_FOLDER ./../../data_dir/Refer_Kitti/KITTI/training/image_02 \
--TRACKERS_FOLDER ./../../track-dataset/gt_template \
--GT_LOC_FORMAT {gt_folder}{video_id}/{expression_id}/gt.txt \
--TRACKERS_TO_EVAL <submit output folder> \
--USE_PARALLEL True \
--NUM_PARALLEL_CORES 2 \
--PLOT_CURVES False \

```

## Acknowledgment

- [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR)
- [TrackEval](https://github.com/JonathonLuiten/TrackEval)
- [MeMOTR](https://github.com/MCG-NJU/MeMOTR/tree/main)
- [iKUN](https://github.com/dyhBUPT/iKUN)