suppressPackageStartupMessages(library(lme4))

arguments <- commandArgs(trailingOnly=FALSE)
file_argument <- sub("^--file=", "", arguments[grepl("^--file=", arguments)])
script_directory <- dirname(normalizePath(file_argument))
input_path <- file.path(script_directory, "results", "speaker_identity_results.csv")

analysis_data <- read.csv(
    input_path,
    stringsAsFactors=FALSE,
    na.strings=c("", "NA")
)

analysis_data$condition <- trimws(analysis_data$condition)
analysis_data$response <- tolower(trimws(analysis_data$response))
analysis_data$model <- tolower(trimws(analysis_data$model))
analysis_data$group <- tolower(trimws(analysis_data$group))

analysis_data <- analysis_data[
    analysis_data$condition %in% c("output_vs_timbreRef", "output_vs_sourceRef") &
    analysis_data$response %in% c("yes", "no") &
    analysis_data$model %in% c("openvoice", "seed_vc") &
    analysis_data$group %in% c("top5", "bottom5"),
]

analysis_data <- analysis_data[!duplicated(analysis_data),]
analysis_data$same_speaker <- as.integer(analysis_data$response == "yes")
analysis_data$model <- factor(
    analysis_data$model,
    levels=c("openvoice", "seed_vc"),
    labels=c("OpenVoice", "SeedVC")
)
analysis_data$tier <- factor(
    analysis_data$group,
    levels=c("bottom5", "top5"),
    labels=c("Bottom 5", "Top 5")
)
analysis_data$condition <- factor(
    analysis_data$condition,
    levels=c("output_vs_sourceRef", "output_vs_timbreRef"),
    labels=c("Output-Style", "Output-Timbre")
)
analysis_data$participant <- factor(analysis_data$participant)
analysis_data$item <- interaction(
    analysis_data$model,
    analysis_data$timbre,
    analysis_data$source,
    drop=TRUE
)

# Positive slopes indicate more same-speaker judgments for the Voice reference
# (timbre) than for the Accent reference (source/style).
analysis_data$voice_vs_accent <- ifelse(
    analysis_data$condition == "Output-Timbre", 0.5, -0.5
)
analysis_data$model_tier <- interaction(
    analysis_data$model, analysis_data$tier, sep=" / ", drop=TRUE
)
analysis_data$seedvc <- as.integer(analysis_data$model == "SeedVC")
analysis_data$seedvc_voice_vs_accent <-
    analysis_data$seedvc * analysis_data$voice_vs_accent

speaker_identity_model <- glmer(
    same_speaker ~ 0 + model_tier + model_tier:voice_vs_accent +
        (1 + voice_vs_accent | participant) +
        (1 + voice_vs_accent | item),
    data=analysis_data,
    family=binomial,
    control=glmerControl(
        optimizer="bobyqa",
        optCtrl=list(maxfun=200000)
    )
)

cat("WITHIN EACH MODEL AND TIER: Voice minus Accent (log odds)\n")
print(summary(speaker_identity_model))

# Equivalent parameterization for testing SeedVC's additional separation.
speaker_identity_comparison_model <- update(
    speaker_identity_model,
    . ~ 0 + model_tier + tier:voice_vs_accent +
        tier:seedvc_voice_vs_accent +
        (1 + voice_vs_accent | participant) + (1 + voice_vs_accent | item)
)
cat("\nBETWEEN MODELS: seedvc_voice_vs_accent tests SeedVC's gap minus OpenVoice's\n")
print(summary(speaker_identity_comparison_model))

# The claim that SeedVC receives more Yes responses on Output-Voice trials
# is distinct from the claim that it has a larger Voice-minus-Accent gap.
speaker_identity_pairing_model <- update(
    speaker_identity_model,
    . ~ 0 + tier:condition + tier:condition:seedvc +
        (1 + voice_vs_accent | participant) + (1 + voice_vs_accent | item)
)
cat("\nWITHIN EACH PAIRING: seedvc tests SeedVC minus OpenVoice\n")
print(summary(speaker_identity_pairing_model))
cat("\nAll summary p-values are unadjusted Wald tests.\n")
cat("\nSingular fit:", isSingular(speaker_identity_model), "\n")
cat("Observations:", nrow(analysis_data), "\n")
cat("Participants:", nlevels(analysis_data$participant), "\n")
cat("Items:", nlevels(analysis_data$item), "\n")
