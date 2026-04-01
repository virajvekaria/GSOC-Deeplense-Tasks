# Specific Test V: Lens Finding And Data Pipelines

This folder contains my solution for the binary lens-finding task: classify each `3 x 64 x 64` object as either lens or non-lens.

## Files

- `task_5.ipynb`: full data pipeline, training, hard-negative mining, and evaluation
- `best_lens_efficient_model.pth`: best checkpoint from the initial EfficientNet-B0 training phase
- `finetuned_lens_efficient_model.pth`: checkpoint after the hard-negative fine-tuning phase
- `result_1.png`: baseline evaluation plots from the main model
- `result_2.png`: evaluation plots after the hard-negative bootcamp fine-tuning phase
- `result_3.png`: final evaluation plots after test-time augmentation

## Approach

The current notebook uses a transfer-learning binary classifier built on top of a pretrained `EfficientNet-B0` backbone:

- pretrained `EfficientNet-B0` feature extractor from `torchvision`
- replaced classifier head with `Dropout -> Linear(256) -> ReLU -> BatchNorm -> Dropout -> Linear(1)`
- end-to-end fine-tuning with the backbone left trainable

Because the dataset is heavily imbalanced, the notebook uses:

- `BinaryFocalLoss` to emphasize the rare positive lens class
- a two-stage workflow with hard-negative mining
- test-time augmentation at inference

In the current notebook version, the focal-loss configuration is:

- `alpha = 0.50`
- `gamma = 2.0`

## Training Strategy

Stage 1 is standard training on the original train split.

Stage 2 is a "bootcamp" fine-tuning pass:

- run the trained model over the training set
- rank non-lens examples by absolute prediction error
- keep the `1500` hardest non-lenses
- combine them with all lens examples
- fine-tune the model again for `3` epochs with a lower learning rate of `1e-7`

In the saved run, the mined bootcamp split contains:

- `1405` lens examples
- `1500` hard non-lens examples
- `2905` total fine-tuning samples

At test time the notebook averages predictions across:

- original image
- horizontal flip
- vertical flip
- `90` degree rotation

## Dataset Setup

The notebook expects the challenge data in four folders:

- `train_lenses`
- `train_nonlenses`
- `test_lenses`
- `test_nonlenses`

It then creates its own train and validation split from the training portion.

The run stored in the notebook uses:

- `24324` training samples
- `6081` validation samples
- `19650` test samples

## Reported Result

Baseline evaluation with the first saved checkpoint:

- accuracy: `0.99`
- macro precision: `0.70`
- macro recall: `0.96`
- macro F1: `0.78`
- test AUC: `0.9911`

Evaluation after the hard-negative bootcamp fine-tuning phase:

- accuracy: `0.99`
- macro precision: `0.75`
- macro recall: `0.95`
- macro F1: `0.82`
- test AUC: `0.9899`

Final evaluation with the fine-tuned model plus test-time augmentation:

- accuracy: `0.99`
- macro precision: `0.76`
- macro recall: `0.95`
- macro F1: `0.83`
- test AUC: `0.9931`

## Result Preview

Baseline evaluation result:

![Test V result 1](result_1.png)

Bootcamp fine-tuned evaluation result:

![Test V result 2](result_2.png)

Fine-tuned plus TTA result:

![Test V result 3](result_3.png)

## Reproducing

1. Download the dataset and place the four challenge folders under the base directory used in the notebook.
2. Update the dataset path cell in `task_5.ipynb` if needed.
3. Run the notebook top to bottom to train, mine hard negatives, fine-tune, and evaluate.

## Notes

- This solution puts more emphasis on recall and ranking quality than on raw positive-class precision.
- The hard-negative phase and the final TTA pass are the significant contributors to the final reported score.
