# Model summary

## vit_cnn

- input (2, 7, 72, 80) -> output (2, 15, 72, 80)
- parameters: 5,941,199

| module | params |
|---|---|
| embed | 137,984 |
| embed_drop | 0 |
| encoder | 4,738,560 |
| norm | 512 |
| stem | 288,608 |
| decoder | 775,040 |
| head | 495 |

## cnn_unet

- input (2, 7, 72, 80) -> output (2, 15, 72, 80)
- parameters: 5,711,151

| module | params |
|---|---|
| enc0 | 23,952 |
| enc1 | 124,800 |
| enc2 | 498,432 |
| enc3 | 1,992,192 |
| bottleneck | 1,327,872 |
| dec2 | 1,327,872 |
| dec1 | 332,160 |
| dec0 | 83,136 |
| head | 735 |

CNN base channels: 48

Parameter ratio cnn_unet / vit_cnn: 0.961
