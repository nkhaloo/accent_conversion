# Accent experiment model

Run `Rscript experiment/perceptual_analysis/speaker_accent_mixed_model.R` from
the repository root. The script prints two equivalent model summaries.

The outcome remains a binary "same accent" response. Output-Style in the data
is Output-Accent in the write-up; Output-Timbre is Output-Voice.
`accent_vs_voice` is +0.5 for Output-Accent and -0.5 for Output-Voice.

The first parameterization is:

```r
same_accent ~ 0 + model_tier + model_tier:accent_vs_voice +
    (1 + accent_vs_voice | participant) + (1 | item)
```

Each `model_tier:accent_vs_voice` coefficient tests the Accent-minus-Voice
difference in conditional log odds for that model and ranking group. Positive
values indicate the intended direction. The four `model_tier` intercepts are
midpoints of the two trial types' conditional log odds, not performance scores.

The second summary expresses the same fit using SeedVC's Accent-minus-Voice
gap and OpenVoice's additional gap within each tier. Its
`tier:openvoice_accent_vs_voice` coefficients directly test whether OpenVoice
has a larger gap than SeedVC. These are log-odds differences, not percentage
point differences. Both summaries retain all eight fixed-effect parameters and
participant random intercepts and trial-type slopes, plus recording random
intercepts. The identity analysis additionally retains recording slopes.

## Results on the saved data

| Summary coefficient interpreted | Beta | Unadjusted p |
| --- | ---: | ---: |
| OpenVoice bottom 5: Accent minus Voice | 2.4168 | 0.0000168 |
| OpenVoice top 5: Accent minus Voice | 2.6293 | 0.0000040 |
| SeedVC bottom 5: Accent minus Voice | 1.2819 | 0.0224 |
| SeedVC top 5: Accent minus Voice | 0.1547 | 0.7966 |
| Bottom 5: OpenVoice gap minus SeedVC gap | 1.1349 | 0.1457 |
| Top 5: OpenVoice gap minus SeedVC gap | 2.4746 | 0.00242 |

The recording-intercept-only structure was restored after the model with
recording slopes produced a singular fit (recording intercept/slope correlation
-1). The restored fit runs without convergence warnings and is nonsingular;
AIC is 1084.9, with 985 observations, 100 participants, and 20 recordings.
Centered participant slope coding is equivalent to the earlier factor coding;
the reported results are restored to rounding precision. These checks are not
a full model diagnostic assessment.

The p-values are the unadjusted, two-sided Wald p-values from `summary()`.
Reparameterizing does not remove multiple-testing considerations; these tests
were selected after inspecting the data and should be described transparently.

A positive gap tests relative separation. It does not establish both an
absolutely high Output-Accent probability and an absolutely low Output-Voice
probability. Those claims require specified thresholds and separate tests.
A nonsignificant gap also does not establish equivalence between trial types.
Inference concerns the selected top/bottom recordings, not all model outputs.

The binomial mixed-model fitting interface is documented in
[lme4's glmer documentation](https://lme4.github.io/lme4/reference/glmer.html).
