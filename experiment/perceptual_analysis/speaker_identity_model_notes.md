# Speaker identity model

Run `Rscript experiment/perceptual_analysis/speaker_identity_mixed_model.R`.
Random slopes now use the centered numeric `voice_vs_accent` predictor, an
equivalent coding that preserves the fitted results.
The script prints three equivalent parameterizations of the same binomial
logistic mixed model. All retain eight fixed-effect parameters and random
intercepts and trial-type slopes for both participants and recordings.

`voice_vs_accent` is +0.5 for Output-Timbre (Output-Voice in the write-up) and
-0.5 for Output-Style (Output-Accent). The first summary gives a separate
Voice-minus-Accent log-odds difference for each model/tier. The second directly
tests SeedVC's gap minus OpenVoice's gap within each tier. The third tests
SeedVC minus OpenVoice within each tier and trial type.

| Comparison | Beta | Unadjusted p |
| --- | ---: | ---: |
| OpenVoice bottom 5: Voice minus Accent | 2.8384 | 0.0000175 |
| OpenVoice top 5: Voice minus Accent | 1.9539 | 0.00264 |
| SeedVC bottom 5: Voice minus Accent | 3.6291 | 0.0000000583 |
| SeedVC top 5: Voice minus Accent | 6.2707 | < 0.000001 |
| Bottom 5: SeedVC gap minus OpenVoice gap | 0.7907 | 0.39519 |
| Top 5: SeedVC gap minus OpenVoice gap | 4.3168 | 0.0000126 |
| Bottom 5, Output-Voice: SeedVC minus OpenVoice | 1.2930 | 0.07871 |
| Top 5, Output-Voice: SeedVC minus OpenVoice | 3.2423 | 0.0000500 |
| Bottom 5, Output-Accent: SeedVC minus OpenVoice | 0.5024 | 0.32117 |
| Top 5, Output-Accent: SeedVC minus OpenVoice | -1.0745 | 0.03781 |

1851 responses, 100 participants, 20 recordings; AIC 1710.0. Fits converged
without singularity. This is not a full model diagnostic assessment.

All p-values are unadjusted two-sided Wald tests from summary(); presenting
comparisons as coefficients does not remove multiple-testing considerations.
The comparisons were selected after inspecting the data. Positive gaps test
relative separation, not absolute high/low probability thresholds. Conclusions
concern these selected top/bottom recordings rather than all model outputs.
