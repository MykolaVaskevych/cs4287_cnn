# Training Results Summary

## All Runs Comparison

| Run | Epochs | Precision | Recall | mAP50 | mAP50-95 | Val Loss |
|-----|--------|-----------|--------|-------|----------|----------|
| ppe_100 (baseline) | 100 | 0.8860 | 0.7378 | 0.7991 | **0.4987** | 1.2892 |
| ppe_100_batch_8 | 100 | 0.8815 | 0.7379 | 0.8079 | **0.5020** | 1.2606 |
| ppe_100_no_mosaic | 100 | 0.9168 | 0.7385 | 0.8255 | **0.5046** | 1.3057 |
| ppe_750_combined | 521* | 0.9076 | 0.7504 | 0.8268 | **0.5359** | 1.1688 |
| ppe_500 (old/saved) | 500 | 0.9130 | 0.7564 | 0.8156 | **0.5499** | 1.1416 |

*Early stopped at epoch 521 (patience=100)

## Duplicate Runs (Invalid)

These runs show IDENTICAL metrics - they are duplicates of the baseline:
- ppe_100_dropout_0.2 (mAP50-95: 0.4987)
- ppe_100_lr_0.001 (mAP50-95: 0.4987)
- ppe_100_lr_0.02 (mAP50-95: 0.4987)

These need to be rerun properly to test actual effect of dropout and learning rate.

## Key Findings

### Valid Experiments (Ranked by mAP50-95)

1. **ppe_500 (old)**: 0.5499 - BEST OVERALL
2. **ppe_750_combined**: 0.5359 - batch=8, no_mosaic, 521 epochs
3. **ppe_100_no_mosaic**: 0.5046 - Best 100-epoch run
4. **ppe_100_batch_8**: 0.5020 - Slight improvement over baseline
5. **ppe_100 (baseline)**: 0.4987 - Reference point

### Analysis

**Surprising Result**: The ppe_500 run with DEFAULT settings (batch=16, mosaic=1.0) achieved 0.5499 mAP50-95, which is BETTER than the ppe_750_combined (batch=8, mosaic=0.0) that only reached 0.5359.

**Why did combined fail to improve?**
- batch=8 alone: +0.33% over baseline (0.5020 vs 0.4987)
- no_mosaic alone: +0.59% over baseline (0.5046 vs 0.4987)
- Combined with 521 epochs: +3.72% over baseline (0.5359 vs 0.4987)
- But still 2.55% WORSE than default 500 epochs (0.5359 vs 0.5499)

Possible explanations:
1. Batch=8 and no_mosaic may have negative interaction
2. Default mosaic augmentation helps long training runs
3. Smaller batch size might need different learning rate schedule
4. Random seed variation

### Best Practices

For this PPE detection task:
- **Best approach**: Default settings (batch=16, mosaic=1.0) with 500+ epochs
- **Quick runs**: Use no_mosaic for 100 epochs (0.5046 in less time)
- **Avoid**: Combining batch=8 with no_mosaic doesn't compound benefits

### Recommendations for Next Runs

1. **Test batch=8 WITH mosaic** for 500 epochs
   - See if batch=8 helps with default augmentation

2. **Test no_mosaic WITH batch=16** for 500 epochs
   - See if no_mosaic helps with longer training

3. **Properly test dropout and learning rate**
   - Rerun lr_0.001, lr_0.02, dropout_0.2 (current runs are duplicates)

4. **Try yolov8s or yolov8m**
   - Larger model might push past 0.55 mAP50-95

## Performance Breakdown

### mAP50-95 Progress
- Baseline 100 epochs: 49.87%
- Best 100 epochs: 50.46% (+0.59%)
- Combined 521 epochs: 53.59% (+3.72%)
- **Best (default 500): 54.99% (+5.12%)**

### Current Best Model
**runs/saved/ppe_500** with configuration:
- batch: 16
- lr0: 0.01
- mosaic: 1.0
- dropout: 0.0
- epochs: 500

Achieves:
- Precision: 91.30%
- Recall: 75.64%
- mAP50: 81.56%
- mAP50-95: 54.99%
