# Specific Test V: Lens Finding And Data Pipelines

This folder contains my solution for the binary lens-finding task: classify each `3 x 64 x 64` object as either lens or non-lens.

## Files

- `task_5.ipynb`: full data pipeline, training, hard-negative mining, and evaluation
- `best_lens_attention_model.pth`: best checkpoint from the initial training phase
- `finetuned_lens_model.pth`: checkpoint after the hard-negative fine-tuning phase
- `result_1.png`: saved evaluation image from the main model
- `result_2.png`: saved evaluation image from the fine-tuned workflow

## Approach

The main model is a custom CNN with a spatial attention block:

- shallow convolutional feature extractor
- `SpatialAttention` module to highlight informative regions
- deeper convolutional block
- dropout-regularized classifier head

Because the dataset is heavily imbalanced, the notebook uses:

- `BinaryFocalLoss` to emphasize the rare positive lens class
- a two-stage workflow with hard-negative mining
- test-time augmentation at inference

## Training Strategy

Stage 1 is standard training on the original train split.

Stage 2 is a "bootcamp" fine-tuning pass:

- run the trained model over the training set
- rank non-lens examples by absolute prediction error
- keep the `1500` hardest non-lenses
- combine them with all lens examples
- fine-tune the model again with a lower learning rate

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

## Reported Result

Evaluation after the second phase:

- accuracy: `0.94`
- macro precision: `0.57`
- macro recall: `0.90`
- macro F1: `0.60`
- test AUC: `0.9648`

Evaluation with the fine-tuned model plus test-time augmentation:

- accuracy: `0.95`
- macro precision: `0.57`
- macro recall: `0.90`
- macro F1: `0.61`
- test AUC: `0.9650`

The notebook keeps high recall on the rare lens class, which is the main priority in this imbalanced setting.

## Result Preview

Main evaluation result:

![Test V result 1](result_1.png)

Fine-tuned plus TTA result:

![Test V result 2](result_2.png)

## Reproducing

1. Download the dataset and place the four challenge folders under the base directory used in the notebook.
2. Update the dataset path cell in `task_5.ipynb` if needed.
3. Run the notebook top to bottom to train, mine hard negatives, fine-tune, and evaluate.

## Notes

- This solution puts more emphasis on recall and ranking quality than on raw positive-class precision.
- The hard-negative phase is the key part of the pipeline and is what produced the final saved model.
