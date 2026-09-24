# iNatSounds → CNN pipeline

This is the production pipeline for wildlife classification. The older
`dataset_producer.py` / `train_cnn.py` scripts are only a small, farm-sound
pipeline check and must not be used to train the wildlife model.

Run the commands below from `Auralis/backend`. The iNatSounds terms permit
non-commercial research/education use and prohibit redistributing recordings;
do not commit the downloaded audio or trained checkpoint.

## 1. Install the backend ML dependencies

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA GPU, install the CUDA-compatible PyTorch and torchvision wheels
from the official PyTorch selector before (or instead of) the last command.
Confirm that `python -c "import torch; print(torch.cuda.is_available())"`
prints `True` before a long full-dataset run.

## Google Colab alternative

Do not upload the routed `inat_routed` directory to Drive or Colab: the full
dataset is over 100 GB and contains more than 100,000 files. Clone the source
code in Colab, download and route the data directly to `/content` (the fast,
temporary runtime disk), and use Drive only for the checkpoint artifacts.

First push the source-code changes to the project GitHub repository (never the
audio/model data). In a new Colab notebook, choose **Runtime → Change runtime
type → T4 GPU** (or any GPU), then run these cells in order:

```python
# Cell 1: source and dependencies
!git clone https://github.com/ShadowMan18/Auralis---CSE_2-2_Term_Project.git
%cd /content/Auralis---CSE_2-2_Term_Project/Auralis/backend
!pip -q install -r requirements.txt
!nvidia-smi
```

```python
# Cell 2: persistent storage for the small checkpoint only
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/Auralis/models
```

```python
# Cell 3: obtain metadata, build assignments, and route audio locally
%cd /content/Auralis---CSE_2-2_Term_Project/Auralis/backend/processor/audio_processing
!python download_inatsounds.py --split train
!python build_taxonomy_classes.py --train-json inatsounds_downloads/train.json --min-count 10
!python inatsounds_route.py --split train --annotations-json inatsounds_downloads/train.json \
  --assignments species_class_assignments.json --output-dir /content/inat_routed \
  --url official --skip-existing
```

```python
# Cell 4: manifests and GPU training
!python build_window_manifest.py --routed-dir /content/inat_routed --split train \
  --also-split-val --out-dir /content/manifests
!python train_cnn_v2.py --train-manifest /content/manifests/train.jsonl \
  --val-manifest /content/manifests/val.jsonl --epochs 40 --model-arch resnet50 \
  --batch-size 32 --num-workers 2 \
  --model-path /content/drive/MyDrive/Auralis/models/species_cnn_v2.pt
```

Each completed epoch writes
`species_cnn_v2.last.pt` to Drive. If Colab disconnects, reconnect, repeat the
source/dependency/data cells if needed, then resume with:

```python
!python train_cnn_v2.py --train-manifest /content/manifests/train.jsonl \
  --val-manifest /content/manifests/val.jsonl --epochs 40 --model-arch resnet50 \
  --batch-size 32 --num-workers 2 \
  --model-path /content/drive/MyDrive/Auralis/models/species_cnn_v2.pt \
  --resume /content/drive/MyDrive/Auralis/models/species_cnn_v2.last.pt
```

Bring these three small files back to
`backend/processor/audio_processing/` after training:

- `species_cnn_v2.pt`
- `species_cnn_v2.labels.json`
- `/content/inat_routed/class_label_map.json` (place it under local
  `inat_routed/class_label_map.json`; the audio files themselves are not needed
  for prediction)

Google documents that mounted Drive is runtime-dependent and that folders with
very many files can be slow or fail. That is why the dataset belongs in
`/content`, while Drive receives only a few model files.

For a species-only run, replace the class-assignment/routing cell above with
the commands in the species-only section below. Run `!df -h /content` before
the five-copy generator. The raw routed corpus plus 685k generated copies can
exceed a free Colab disk. If storage is tight and the routed source does not
need to be retained, add `--delete-source-after` to the generator command;
each source is removed only after all its generated output files succeed.

## 2. Obtain metadata and choose class resolution

```powershell
python processor/audio_processing/download_inatsounds.py --split train
python processor/audio_processing/build_taxonomy_classes.py --min-count 10
python processor/audio_processing/inatsounds_route.py --split train `
  --annotations-json processor/audio_processing/inatsounds_downloads/train.json `
  --assignments processor/audio_processing/species_class_assignments.json --plan
```

The checked metadata currently yields 136,941 train recordings routed to 2,068
CNN classes. Species with fewer than ten recordings are pooled at genus,
family, or order level; 71 recordings / 33 species remain zero-shot-only.
Species labels include the scientific name to prevent duplicate common names
from merging. Therefore use that exact label in a geo-prior table.

### Species-only mode with five augmented copies per training recording

If your required output is always an individual species rather than a
genus/family/order group, create a separate species-only assignment file. This
creates 5,569 species labels, including the 1,951 species represented by only
one original recording. Five altered copies improve pitch/noise/tempo tolerance
but do not give those labels five independent animal observations.

```powershell
python processor/audio_processing/build_taxonomy_classes.py --species-only `
  --out processor/audio_processing/species_assignments_only.json
```

Route with that assignment file into a distinct folder named
`inat_routed_species`. Then make five modified, fixed three-second training
copies per original recording:

```powershell
python processor/audio_processing/dataset_generator_v2.py `
  --routed-dir processor/audio_processing/inat_routed_species `
  --output-dir processor/audio_processing/inat_augmented_species `
  --variants-per-file 5
```

The generator splits by original recording before augmenting, so a source and
its variants cannot leak into both train and validation. Build the two
manifests separately afterwards (do not add `--also-split-val`):

```powershell
python processor/audio_processing/build_window_manifest.py `
  --routed-dir processor/audio_processing/inat_augmented_species --split train `
  --out-dir processor/audio_processing/manifests_species
python processor/audio_processing/build_window_manifest.py `
  --routed-dir processor/audio_processing/inat_augmented_species --split val `
  --out-dir processor/audio_processing/manifests_species
```

The full five-copy corpus is hundreds of GB. First smoke-test the generator
with `--max-source-files 100`; use Colab `/content` storage only if it has
enough free space.

Start with mammals before committing to a full run:

```powershell
python processor/audio_processing/build_taxonomy_classes.py --min-count 10 `
  --supercategory Mammalia `
  --out processor/audio_processing/species_class_assignments_mammalia.json
```

## 3. Download and route real labelled audio

There are two valid choices. Do not use the `test` split for model selection.

**Recommended when disk is limited:** directly stream the official 81 GB train
archive and write only class-organised audio. A dropped connection requires a
new archive stream, but `--skip-existing` avoids overwriting completed files.

```powershell
python processor/audio_processing/inatsounds_route.py --split train `
  --annotations-json processor/audio_processing/inatsounds_downloads/train.json `
  --assignments processor/audio_processing/species_class_assignments.json `
  --output-dir processor/audio_processing/inat_routed --url official --skip-existing
```

**Recommended when network reliability matters and you have 81+ GB spare:**
download a resumable local archive first, then route it. The downloader resumes
an interrupted `.part` file using HTTP Range.

```powershell
python processor/audio_processing/download_inatsounds.py --split train --audio
python processor/audio_processing/inatsounds_route.py --split train `
  --tar processor/audio_processing/inatsounds_downloads/train.tar.gz `
  --annotations-json processor/audio_processing/inatsounds_downloads/train.json `
  --assignments processor/audio_processing/species_class_assignments.json `
  --output-dir processor/audio_processing/inat_routed --skip-existing
```

For the mammal trial, substitute
`species_class_assignments_mammalia.json` and use a separate output folder such
as `inat_routed_mammalia`.

## 4. Build leak-free window manifests

The following creates 3-second windows every 1.5 seconds and makes the
train/validation split by original recording, never by window.

```powershell
python processor/audio_processing/build_window_manifest.py `
  --routed-dir processor/audio_processing/inat_routed --split train --also-split-val
```

This is preferred over routing `val.tar.gz`: it avoids another 25 GB download.
The generated background recordings are also split by recording and cannot
leak into both splits. Replace them with real forest ambience later via
`--background-dir` for better rejection of wind, rain, insects, and silence.

## 5. Train and evaluate

First make a cheap smoke test on the mammal output (use the corresponding
routed directory):

```powershell
python processor/audio_processing/train_cnn_v2.py `
  --epochs 10 --model-arch resnet18 --batch-size 16 --num-workers 0 `
  --model-path processor/audio_processing/species_cnn_mammalia.pt
```

Then train the full model. `--num-workers 0` is safest on Windows; raise it
only after a successful run. The default soft class balancing prevents the
huge taxonomic rollups from dominating every minibatch.

```powershell
python processor/audio_processing/train_cnn_v2.py `
  --epochs 40 --model-arch resnet50 --batch-size 32 --num-workers 0 `
  --model-path processor/audio_processing/species_cnn_v2.pt
```

Keep the checkpoint with its automatically written
`species_cnn_v2.labels.json`. A validation score is not enough: test recordings
from unseen sites/devices and inspect top-5 predictions, especially for your
target region. The model may return an explicitly marked genus/family/order
group where the dataset does not support species-level identification.

## 6. Serve predictions with location and season

`/api/upload-sample` now uses the phase-2 CNN. It accepts multipart fields:
`sample`, optional `lat`, `lon`, `month` (1–12), and optional `top_k`.
Location fields must be provided together. A region name needs to be resolved
to coordinates before calling the API.

Optionally set these in backend `.env` before starting Flask:

```text
AURALIS_CNN_MODEL_PATH=processor/audio_processing/species_cnn_v2.pt
AURALIS_GEO_PRIOR_JSON=path/to/region_season_prior.json
```

The geo-prior JSON uses the format documented in `geo_prior.py`. Use canonical
species labels such as `Jaguar (Panthera onca)`, not a bare potentially
ambiguous common name. Without `AURALIS_GEO_PRIOR_JSON`, location data is
accepted but the response correctly remains CNN-only.

The response contains a `predictions` list of `{species, confidence}` plus the
legacy parallel `species` / `confidence` lists. It returns HTTP 503 until both
the checkpoint and labels JSON exist, rather than silently falling back to the
old farm-animal fingerprint matcher.
