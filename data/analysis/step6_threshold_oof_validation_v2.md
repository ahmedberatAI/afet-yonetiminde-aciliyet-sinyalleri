# Step 6 OOF Validation v2

## exp1_gold_v2_bce
- rows: 1740
- global 0.5 -> micro=0.775, macro=0.342, P=0.907, R=0.677
- cv tuned -> micro=0.740, macro=0.448, P=0.688, R=0.799
- delta -> micro=-0.036, macro=+0.106
- rare-label OOF F1 (tuned): altyapi=0.082, guvenlik=0.056, psikolojik=0.000

## exp2_gold_v2_posw
- rows: 1740
- global 0.5 -> micro=0.667, macro=0.464, P=0.544, R=0.862
- cv tuned -> micro=0.726, macro=0.523, P=0.696, R=0.757
- delta -> micro=+0.059, macro=+0.059
- rare-label OOF F1 (tuned): altyapi=0.148, guvenlik=0.644, psikolojik=0.000

