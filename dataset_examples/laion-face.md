# Laion-Face

[LAION-Face](https://github.com/FacePerceiver/LAION-Face) is the human face subset of [LAION-400M](https://laion.ai/laion-400-open-dataset/), it consists of 50 million image-text pairs. Face detection is conducted to find images with faces. Apart from the 50 million full-set(LAION-Face 50M), there is a 20 million sub-set(LAION-Face 20M) for fast evaluation. 

LAION-Face is first used as the training set of [FaRL](https://github.com/FacePerceiver/FaRL), which provides powerful pre-training transformer backbones for face analysis tasks.

For more details, please check the offical repo at https://github.com/FacePerceiver/LAION-Face .

## Download and convert metadata
```bash
wget -l1 -r --no-parent https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/
mv the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/ .
wget https://huggingface.co/datasets/FacePerceiver/laion-face/resolve/main/laion_face_ids.pth
wget https://raw.githubusercontent.com/FacePerceiver/LAION-Face/master/convert_parquet.py
python convert_parquet.py ./laion_face_ids.pth ./laion400m-meta ./laion_face_meta
```

## Download the images with img2dataset
When metadata is ready, you can start download the images.

```bash
wget https://raw.githubusercontent.com/FacePerceiver/LAION-Face/master/download.sh
bash download.sh ./laion_face_meta ./laion_face_data
```

Please be patient, this command might run over days, and cost about 2T disk space, and it will download 50 million image-text pairs as 32 parts.

- To use the **LAION-Face 50M**, you should use all the 32 parts.
- To use the **LAION-Face 20M**, you should use these parts.
    ```
    0,2,5,8,13,15,17,18,21,22,24,25,28
    ```

checkout `download.sh` and [img2dataset](https://github.com/rom1504/img2dataset) for more details and parameter setting.



