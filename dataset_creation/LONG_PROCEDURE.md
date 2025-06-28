# Creating the AI-GenBench dataset: long version

Beware that this readme covers the long procedure to create the AI-GenBench dataset. You are probably looking for the simple version: please refer to the main [README](README.md).

## Important
- Beware that **getting things right on the first attempt is not easy**, so please open a discussion or an issue if you need help! 
- Also consider that 4TB of storage and 64GB of RAM are recommended for the dataset creation process. The storage is required to keep the origin datasets plus the intermediate and final outputs of the script. The dataset creation process is internally parallelized, so using a NVME Gen4+ SSD is highly recommended. The dataset creation process will take a long time, so be patient.
Note: it is totally doable to create the dataset on a slow / network attached storage: this is how we built it on our side :). If running over NFS, consider NFSv4 with hard mount, or any NFS that supports memory mapping. Do not use SMB.
- Internal parallelism is achieved using the `dataset`, `multiprocessing`, and `joblib` libraries, and this may be a problem when using Windows. Consider using a Linux-based system for the dataset creation process. If you are using Windows, consider using WSL2.
- A part of the real datasets (LAION-400M) is scraped, so it may not be possible to re-create the dataset exactly as it was firstly created. This is not an issue when submitting for having your method included in the leaderboard!
- Read through the creation steps and the dataset description before starting. The dataset is large and the creation process is long, so it is better to understand what you are doing before starting.

## Getting started
1. Clone this repository.
2. Install the required dependencies. You can either install the reduced dataset-build only dependencies or the training and evaluation dependencies as well.
    ```bash
    pip install -r requirements_dataset_creation.txt
    ```
    or:
    ```bash
    pip install -r requirements_train.txt
    ```

## Dataset creation steps
0. The `7zz` executable must be available in the PATH.
  - Recommended: download the latest 64-bit Linux package from the [official 7-zip website](https://www.7-zip.org/download.html), unzip it, and add that folder to your PATH environment variable. Consider not using the version that comes with p7zip as, depending on the distribution, is *usually* quite an old version.
1. Download the datasets (download links below).
    - Note: some datasets must be scraped/downloaded from a file list. There are some [img2dataset](https://github.com/rom1504/img2dataset)-based scripts in [`dataset_download_scripts`](dataset_download_scripts) for those datasets (you will need to download the file list manually, though).
    - Consider storing the datasets (and intermediate files, see below) on a fast and high-capacity SSD.
    - Note: in this version, **you don't have to extract GenImage or DRCT zips**. Just keep them zipped; images will be taken from those archives using `7z`. This will be extended to other datasets in future versions (just a matter of coding it, you are welcome to help!). Of course, if you have a pre-unzipped version of those datasets you can use it as long you didn't change the original folder structure.
2. Create a `local.cfg` in the project root with this minimal structure:
    ```
    [paths]
    input_datasets_path = /downloaded_datasets
    output_path = /outputs/deepfake_dataset
    tmp_cache_dir = /outputs/staging_area/volatile_cache
    intermediate_outputs_path = /outputs/staging_area/intermediate_datasets
    ```
    - `input_datasets_path` is where you put the datasets you just downloaded.
    - `output_path` is where the final dataset will be saved.
    - `tmp_cache_dir` is a temporary directory used by the `datasets` library during the generation of datasets, usually automatically emptied once the main script is done.
    - `intermediate_outputs_path` is where intermediate subsets will be stored. It is a cache and is used in successive runs of the main script.
3. Add custom paths to `local.cfg`, such as:
   ```
   imagenet = /path/to/imagenet
   coco = /path/to/coco
   ```
   This will allow you to have datasets in different paths (by default, they are looked for in `input_datasets_path`). See the [dataset_paths.py](dataset_paths.py) file for a complete list of dataset keys.
4. Run the `main_dataset_generation.py` script (this will take quite some time).
5. Once the dataset is created, you will need a final step. Alas, the dataset you'll find in `output_path` is based on `LargeImage`, a custom feature we needed to implement to circumvent various long-standing issues in the `datasets` library. To use the dataset, either:
    - Import and register `LargeImage` in all codebases that will use the datasets (may make sense for most use cases):
      ```python
      from custom_features.large_image import LargeImage
      datasets.features.features.register_feature(LargeImage, "LargeImage")
      ```
    - Or, if you need to use the dataset in a different way, you can convert it to a standard `Image` dataset. To do this, run:
      ```bash
      cd dataset_preparation
      python finalize_dataset.py \
        --dataset_path /path/to/your/dataset \
        --output_path /path/to/your/final_dataset
      ```
      Finalizing is mostly dependent on storage speed.
    - We are trying to remove this additional step, but it is not easy. The `datasets` library has some long-standing issues with large datasets, and we are trying to work around them. If you have any suggestions or ideas, please let us know. Main issues are: https://github.com/huggingface/datasets/issues/5717, https://github.com/huggingface/datasets/issues/615. Some issues have been circumvented by enforcing a shard size, but that was not enough. For LargeImage followed the advice here: https://github.com/huggingface/datasets/issues/5717#issuecomment-2459959869.

## Origin datasets
These are the datasets used to create the AI-GenBench dataset. For instructions on how to download them, refer to the [Download links](#download-links) section. Also, make sure you understand the related licensing terms that can be found on their websites.

### Synthetic images
- [Aeroblade](https://github.com/jonasricker/aeroblade)
- [Artifact](https://github.com/awsaf49/artifact/)
- [DDMD (Towards the Detection of Diffusion Model Deepfakes)](https://github.com/jonasricker/diffusion-model-deepfake-detection)
- [DMimageDetection](https://github.com/grip-unina/DMimageDetection)
- [DRCT-2M](https://github.com/beibuwandeluori/DRCT)
- [ELSA_D3](https://huggingface.co/datasets/elsaEU/ELSA_D3)
- [SFHQ-T2I](https://github.com/SelfishGene/SFHQ-T2I-dataset)
- [Forensynths](https://peterwang512.github.io/CNNDetection)
- [GenImage](https://genimage-dataset.github.io)
- [Imaginet](https://github.com/delyan-boychev/imaginet)
- [Polardiffshield](https://github.com/qbammey/polardiffshield)
- [Synthbuster](https://www.veraai.eu/posts/dataset-synthbuster-towards-detection-of-diffusion-model-generated-images)

### Real images
- The part of the [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) dataset used in [ELSA_D3](https://huggingface.co/datasets/elsaEU/ELSA_D3)
- [COCO 2017 train and validation](https://cocodataset.org)
- [ImageNet (ILSVRC 2012)](https://www.image-net.org/index.php)
- [RAISE (all)](http://loki.disi.unitn.it/RAISE/index.php)

## Generators breakdown
For the appropriate dataset / paper citation, please refer to the list in Table 2 of [our paper](https://arxiv.org/abs/2504.20865). For a more precise breakdown of the origin of the images used in the dataset, please refer to the [`resources/DATASET_STATS.md`](resources/DATASET_STATS.md) file.

- [**ADM** (a.k.a. Guided Diffusion)](https://github.com/openai/guided-diffusion), from:
  - DDMD
  - DMimageDetection
  - GenImage
- [**BigGAN**](https://arxiv.org/abs/1809.11096v2), from:
  - Artifact
  - DMimageDetection
  - Forensynths
- [**CIPS**](https://arxiv.org/abs/2011.13775), from:
  - Artifact
- [**Cascaded Refinement Networks**](https://arxiv.org/abs/1707.09405), from:
  - Forensynths
- [**CycleGAN**](https://arxiv.org/abs/1703.10593), from:
  - Artifact
  - Forensynths
- **DALL-E**, from:
  - Imaginet
  - Polardiffshield
  - Synthbuster
- [**DDPM**](https://arxiv.org/abs/2006.11239), from:
  - Artifact
  - DDMD
- [**DeepFloyd-IF**](https://github.com/deep-floyd/IF), from:
  - ELSA_D3
- [**Denoising Diffusion GAN**](https://arxiv.org/abs/2112.07804), from:
  - Artifact
- [**Diffusion GAN**](https://arxiv.org/abs/2206.02262) (ProjectedGAN and StyleGAN), from:
  - Artifact
  - DDMD
- **FLUX 1 dev**, from:
  - FHQ-T2I
- **FLUX 1 schnell**, from:
  - FHQ-T2I
- [**FaceSynthetics**](https://arxiv.org/abs/2109.15102), from:
  - Artifact
- [**GANformer**](https://arxiv.org/abs/2111.08960), from:
  - Artifact
- [**GauGAN (a.k.a. SPADE)**](https://github.com/NVlabs/SPADE), from:
  - Artifact
  - Forensynths
- [**Glide**](https://arxiv.org/abs/2112.10741), from:
  - Artifact
  - DMimageDetection
  - GenImage
  - Polardiffshield
  - Synthbuster
- [**IMLE**](https://www.sfu.ca/~keli/projects/imle/scene_layouts/), from:
  - Forensynths
- [**LaMa**](https://arxiv.org/abs/2109.07161), from:
  - Artifact
- [**LatentDiffusion**](https://github.com/CompVis/latent-diffusion), from:
  - Artifact
  - DDMD
  - DMimageDetection
- [**MAT**](https://arxiv.org/abs/2203.15270), from:
  - Artifact
- [**Midjourney**](https://aituts.com/midjourney-versions/) (various versions), from:
  - GenImage
- [**Palette**](https://arxiv.org/abs/2111.05826), from:
  - Artifact
- [**ProGAN**](https://github.com/tkarras/progressive_growing_of_gans), from:
  - Artifact
  - DDMD
  - DMimageDetection
  - Forensynths
- [**ProjectedGAN**](https://arxiv.org/abs/2111.01007), from:
  - Artifact
  - DDMD
- [**SN-PatchGAN**](https://arxiv.org/abs/1806.03589), from:
  - Artifact
- [**Stable Diffusion**](https://ommer-lab.com/research/latent-diffusion-models/) (mix of *undefined versions*), from:
  - DMimageDetection
  - Artifact
- **Stable Diffusion 1.4**, from:
  - ELSA_D3
  - GenImage
  - Polardiffshield
  - Synthbuster
- **Stable Diffusion 1.5**, from:
  - Aeroblade
  - GenImage
- **Stable Diffusion 2.1**, from:
  - Aeroblade
  - ELSA_D3
  - Imaginet
  - Polardiffshield
- **Stable Diffusion XL 1.0**, from:
  - ELSA_D3
  - Imaginet
  - Polardiffshield
  - Synthbuster
- [**StarGAN v1**](https://arxiv.org/abs/1711.09020), from:
  - Artifact
  - Forensynths
- [**StyleGAN1**](https://arxiv.org/abs/1812.04948), from:
  - Artifact
  - DDMD
  - Forensynths
- [**StyleGAN2**](https://arxiv.org/abs/1912.04958), from:
  - Artifact
  - DMimageDetection
  - Forensynths
- [**StyleGAN2 (SFHQ)**](https://github.com/SelfishGene/SFHQ-dataset), from:
  - Artifact
- [**StyleGAN3**](https://arxiv.org/abs/2106.12423), from:
  - Artifact
  - DMimageDetection
  - Imaginet
- [**VQ-Diffusion**](https://arxiv.org/abs/2111.14822), from:
  - Artifact
  - GenImage
- [**VQGAN**](https://arxiv.org/abs/2012.09841) (a.k.a. Taming Transformers), from:
  - Artifact
  - DMimageDetection
- [**Wukong**](https://xihe.mindspore.cn/modelzoo/wukong), from:
  - GenImage
    - More info: https://xihe.mindspore.cn/modelzoo/wukong/introduce
    - Note: for Wukong, the positive prompt included in the dataset is in English, but this is not the actual prompt used by the GenImage authors (they translated the prompt to Chinese to get better performance).
    For a possible translation of ImageNet class names, please refer to:
        - https://github.com/ningbonb/imagenet_classes_chinese/blob/master/imagenet_classes.js
        - https://github.com/gregor-ge/Babel-ImageNet/blob/main/data/babel_imagenet.json

Plase note that not all the generators are used in the final benchmark. Only the ones in the list reported in the [AI-GenBench paper](https://arxiv.org/abs/2504.20865) are used for the benchmark. The rest of the generators are included in the dataset for completeness, but are ignored from the training/eval codebase.

## Download links

Note: you will need to download these manually and put them in a folder of your choice (and then set it to `local.cfg`). For *some* datasets (Polardiffshield, the LAION subset, and RAISE), a download script is available in the [`dataset_download_scripts`](dataset_download_scripts) (you will still need the file list).

IMPORTANT: if you find issues downloading those datasets, please open an issue or a discussion. We will try to help you. Also, if you find a better download link/mirror, please let us know and we will update the list.

### Datasets of fake images
- [Aeroblade](https://zenodo.org/records/10997235): manual download
- [Artifact](https://www.kaggle.com/datasets/awsaf49/artifact-dataset): manual download
- [DDMD (Towards the Detection of Diffusion Model Deepfakes)](https://zenodo.org/records/7528113): manual download
- [DMimageDetection](https://www.grip.unina.it/download/prog/DMimageDetection/latent_diffusion_trainingset.zip): manual download
  - Note: the GAN part of the training set is the same as the Forensynths one (not included in the given link).
- [DRCT-2M](https://modelscope.cn/datasets/BokingChen/DRCT-2M/files)
  - Note: you don't need extract DRCT zips (see step 0 of the dataset creation process).
- [ELSA_D3](https://huggingface.co/datasets/elsaEU/ELSA_D3): download using [datasets](https://huggingface.co/docs/datasets/en/loading#hugging-face-hub)
  - **Important**: store it offline using [`save_to_disk`](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset.save_to_disk)
- [SFHQ-T2I](https://www.kaggle.com/datasets/selfishgene/sfhq-t2i-synthetic-faces-from-text-2-image-models): manual download
- [Forensynths](https://drive.google.com/drive/folders/1RwCSaraEUctIwFgoQXWMKFvW07gM80_3): manual download
  - Note: we found its download links to be quite unstable. Consider checking the [original repository](https://github.com/peterwang512/CNNDetection) for updated links.
- [GenImage](https://drive.google.com/drive/folders/1jGt10bwTbhEZuGXLyvrCuxOI0cBqQ1FS): manual download
  - Note: you don't have to extract GenImage zips (see step 0 of the dataset creation process).
- [Imaginet](https://huggingface.co/datasets/delyanboychev/imaginet): download using [datasets]
  - **Important**: store it offline using [`save_to_disk`](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset.save_to_disk)
- [Polardiffshield](https://github.com/qbammey/polardiffshield): 
  - Download the filelist, and then check [`dataset_download_scripts`](dataset_download_scripts) for the appropriate script to download images.
- [Synthbuster](https://www.veraai.eu/posts/dataset-synthbuster-towards-detection-of-diffusion-model-generated-images) : manual download

### Datasets of real images
- The part of the [LAION-400M](https://laion.ai/blog/laion-400-open-dataset/) dataset used in [ELSA_D3](https://huggingface.co/datasets/elsaEU/ELSA_D3)
  - **You need to download `ELSA_D3` first!**
  - The filelist is already inside ELSA_D3.
  - Once ELSA_D3 is ready, use the appropriate script in [`dataset_download_scripts`](dataset_download_scripts) to download the real images.
  - Note: the LAION dataset is scraped from the internet. This may be the only source of problems as some images may not be available anymore. Because of this, you may not be able to re-create the *exact* dataset we used: this is not a big issue as the dataset is quite large the script will just use some other images instead.
- [COCO 2017 train and validation](https://cocodataset.org): manual download
- [ImageNet (ILSVRC 2012)](https://www.image-net.org/index.php): manual download
- [RAISE (all)](http://loki.disi.unitn.it/RAISE/index.php)
  - Download the filelist, and then check [`dataset_download_scripts`](dataset_download_scripts) for the appropriate script to download images.

## Leaderboard and paper
- Our paper, **AI-GenBench: A New Ongoing Benchmark for AI-Generated Image Detection**, is available on:
    - IJCNN 2025 proceedings (Verimedia workshop) (to be published)
    - [arXiv](https://arxiv.org/abs/2504.20865)
- For an up-to-date leaderboard of the benchmark, please refer to the [README in the root of the repository](../README.md)
