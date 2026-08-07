suppressPackageStartupMessages(library(lme4))

arguments <- commandArgs(trailingOnly=FALSE)
file_argument <- sub("^--file=", "", arguments[grepl("^--file=", arguments)])
script_directory <- dirname(normalizePath(file_argument))
input_path <- file.path(script_directory, "results", "speaker_accent_results.csv")

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
analysis_data$same_accent <- as.integer(analysis_data$response == "yes")
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

speaker_accent_model <- glmer(
    same_accent ~ model * tier * condition +
        (1 + condition | participant) +
        (1 | item),
    data=analysis_data,
    family=binomial,
    control=glmerControl(
        optimizer="bobyqa",
        optCtrl=list(maxfun=200000)
    )
)

print(summary(speaker_accent_model))
cat("\nSingular fit:", isSingular(speaker_accent_model), "\n")
cat("Observations:", nrow(analysis_data), "\n")
cat("Participants:", nlevels(analysis_data$participant), "\n")
cat("Items:", nlevels(analysis_data$item), "\n")
