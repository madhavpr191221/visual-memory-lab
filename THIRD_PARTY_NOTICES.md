# Third-Party Notices

This repository contains original research code and documentation that depend
on third-party datasets, models, libraries, and services. Third-party materials
remain subject to their own licenses and terms.

## Microsoft Research 7-Scenes

The Phase 3 and Phase 4 experiments use the Office scene from the Microsoft
Research RGB-D Dataset 7-Scenes.

- Dataset page and license:
  <https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/>
- Original paper: Jamie Shotton, Ben Glocker, Christopher Zach, Shahram Izadi,
  Antonio Criminisi, and Andrew Fitzgibbon. “Scene Coordinate Regression
  Forests for Camera Relocalization in RGB-D Images.” CVPR, 2013.
- Publication page:
  <https://www.microsoft.com/en-us/research/publication/scene-coordinate-regression-forests-for-camera-relocalization-in-rgb-d-images-2/>

Microsoft Research provides 7-Scenes for non-commercial use. This repository
uses it only for non-commercial research, public demonstration, and personal
portfolio review. It does not redistribute the original RGB images, depth
images, poses, scene archives, embeddings, or reconstructions.

The tracked `artifacts/phase3/office-zones.json` file is a modified,
dataset-derived annotation artifact created on August 11, 2026. It contains
observation identifiers, machine-assisted semantic zone definitions, and
assignments; it does not contain the source RGB-D frames. The artifact is made
available only for the same non-commercial research and demonstration purpose,
subject to the applicable Microsoft Research License Agreement. See
[`artifacts/phase3/NOTICE.md`](artifacts/phase3/NOTICE.md).

Microsoft provides its materials as-is and disclaims warranties and liability
as described in the complete license agreement linked from the dataset page.

## OpenAI CLIP

The project uses the `openai/clip-vit-base-patch32` model through Hugging Face
Transformers. The repository does not redistribute CLIP weights.

- Alec Radford et al. “Learning Transferable Visual Models From Natural
  Language Supervision.” ICML, 2021:
  <https://proceedings.mlr.press/v139/radford21a.html>
- Official implementation: <https://github.com/openai/CLIP>
- MIT License: <https://github.com/openai/CLIP/blob/main/LICENSE>
- Model card: <https://github.com/openai/CLIP/blob/main/model-card.md>

The model card frames CLIP as a research output, documents its limitations, and
places surveillance and facial-recognition uses outside its intended scope.
This project does not perform person identification or facial recognition.

## OpenAI API outputs

The VLM-assisted zone descriptions and optional evidence analyses were created
using the OpenAI API. API use and generated outputs remain subject to the
[OpenAI terms and policies](https://openai.com/policies/). No API key is stored
in this repository.

## Package dependencies

Python and JavaScript dependencies are declared in `pyproject.toml`, `uv.lock`,
`web/package.json`, and `web/package-lock.json`. They are not relicensed by this
project and remain subject to their respective upstream licenses.
