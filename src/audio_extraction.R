library(tidyverse)
library(rvest)
library(xml2)
library(fs)
library(glue)

base_url <- "https://accent.gmu.edu"
browse_url <- paste0(base_url, "/browse_language.php")

page <- read_html(browse_url)

links_tbl <- tibble(
  text = page %>% html_elements("a") %>% html_text2(),
  href = page %>% html_elements("a") %>% html_attr("href")
) %>%
  mutate(
    text_lower = str_to_lower(text),
    href = if_else(is.na(href), "", href)
  )

language_pages <- links_tbl %>%
  filter(text_lower %in% c("gujarati", "hindi")) %>%
  mutate(full_url = paste0(base_url, "/", href))

get_speaker_links <- function(language_url) {
  lang_page <- read_html(language_url)

  tibble(
    speaker_text = lang_page %>% html_elements("a") %>% html_text2(),
    href = lang_page %>% html_elements("a") %>% html_attr("href")
  ) %>%
    mutate(href = if_else(is.na(href), "", href)) %>%
    filter(str_detect(href, "speakerid=")) %>%
    mutate(full_url = paste0(base_url, "/", href))
}

speaker_links <- language_pages %>%
  rename(language = text) %>%
  mutate(speaker_tbl = map(full_url, get_speaker_links)) %>%
  select(language, speaker_tbl) %>%
  unnest(speaker_tbl)

extract_field <- function(text, field_name) {
  pattern <- paste0(field_name, ":\\s*(.*)")
  match <- str_match(text, pattern)
  match[, 2]
}

process_speaker <- function(speaker_url, speaker_name, language, base_url,
                            audio_dir = "reference") {
  Sys.sleep(0.3)

  speaker_page <- read_html(speaker_url)

  page_text <- speaker_page %>%
    html_element("body") %>%
    html_text2()

  audio_candidates <- c(
    speaker_page %>% html_elements("audio") %>% html_attr("src"),
    speaker_page %>% html_elements("source") %>% html_attr("src"),
    speaker_page %>% html_elements("a") %>% html_attr("href")
  )

  audio_candidates <- audio_candidates[!is.na(audio_candidates)]
  mp3_url <- audio_candidates[str_detect(audio_candidates, "\\.mp3")][1]

  if (!is.na(mp3_url) && !str_detect(mp3_url, "^https?://")) {
    mp3_url <- paste0(base_url, "/", mp3_url) %>%
      str_replace_all("/+", "/") %>%
      str_replace("https:/", "https://")
  }

  mp3_file <- NA_character_

  if (!is.na(mp3_url)) {
    speaker_id <- str_extract(speaker_url, "\\d+")

    mp3_file <- file.path(
      audio_dir,
      paste0(speaker_name, "_", speaker_id, ".mp3")
    )

    mp3_file <- str_replace_all(mp3_file, "[^[:alnum:]_./-]", "_")

    if (!file.exists(mp3_file)) {
      download.file(mp3_url, destfile = mp3_file, mode = "wb", quiet = TRUE)
    }
  }

  tibble(
    language = language,
    speaker_name = speaker_name,
    speaker_url = speaker_url,
    birth_place = extract_field(page_text, "birth place"),
    native_language = extract_field(page_text, "native language"),
    other_languages = extract_field(page_text, "other language\\(s\\)"),
    age_sex = extract_field(page_text, "age, sex"),
    age_onset = extract_field(page_text, "age of english onset"),
    learning_method = extract_field(page_text, "english learning method"),
    english_residence = extract_field(page_text, "english residence"),
    length_residence = extract_field(page_text, "length of english residence"),
    mp3_url = mp3_url,
    mp3_file = mp3_file
  )
}

audio_dir <- "reference"
data_dir <- "data"

dir.create(audio_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

all_speakers_df <- pmap_dfr(
  list(
    speaker_links$full_url,
    speaker_links$speaker_text,
    speaker_links$language
  ),
  \(speaker_url, speaker_name, language) {
    message("Processing: ", speaker_name, " (", language, ")")

    tryCatch(
      process_speaker(
        speaker_url = speaker_url,
        speaker_name = speaker_name,
        language = language,
        base_url = base_url,
        audio_dir = audio_dir
      ),
      error = function(e) {
        message("FAILED: ", speaker_name, " | URL: ", speaker_url)
        message("ERROR: ", conditionMessage(e))

        tibble(
          language = language,
          speaker_name = speaker_name,
          speaker_url = speaker_url,
          birth_place = NA_character_,
          native_language = NA_character_,
          other_languages = NA_character_,
          age_sex = NA_character_,
          age_onset = NA_character_,
          learning_method = NA_character_,
          english_residence = NA_character_,
          length_residence = NA_character_,
          mp3_url = NA_character_,
          mp3_file = NA_character_
        )
      }
    )
  }
)

csv_path <- file.path(data_dir, "accent_archive_metadata.csv")
write_csv(all_speakers_df, csv_path)

print(csv_path)
print(nrow(all_speakers_df))


# ----------------------------
# English source speakers
# ----------------------------

source_dir <- "source"
dir.create(source_dir, showWarnings = FALSE, recursive = TRUE)

english_url <- paste0(base_url, "/browse_language.php?function=find&language=english")

english_speaker_links <- get_speaker_links(english_url) %>%
  mutate(language = "english")

process_speaker_metadata_only <- function(speaker_url, speaker_name, language, base_url) {
  Sys.sleep(0.3)

  speaker_page <- read_html(speaker_url)

  page_text <- speaker_page %>%
    html_element("body") %>%
    html_text2()

  audio_candidates <- c(
    speaker_page %>% html_elements("audio") %>% html_attr("src"),
    speaker_page %>% html_elements("source") %>% html_attr("src"),
    speaker_page %>% html_elements("a") %>% html_attr("href")
  )

  audio_candidates <- audio_candidates[!is.na(audio_candidates)]
  mp3_url <- audio_candidates[str_detect(audio_candidates, "\\.mp3")][1]

  if (!is.na(mp3_url) && !str_detect(mp3_url, "^https?://")) {
    mp3_url <- paste0(base_url, "/", mp3_url) %>%
      str_replace_all("/+", "/") %>%
      str_replace("https:/", "https://")
  }

  tibble(
    language = language,
    speaker_name = speaker_name,
    speaker_url = speaker_url,
    birth_place = extract_field(page_text, "birth place"),
    native_language = extract_field(page_text, "native language"),
    other_languages = extract_field(page_text, "other language\\(s\\)"),
    age_sex = extract_field(page_text, "age, sex"),
    age_onset = extract_field(page_text, "age of english onset"),
    learning_method = extract_field(page_text, "english learning method"),
    english_residence = extract_field(page_text, "english residence"),
    length_residence = extract_field(page_text, "length of english residence"),
    mp3_url = mp3_url
  )
}

download_selected_speaker <- function(speaker_name, speaker_url, mp3_url, audio_dir = "source") {
  if (is.na(mp3_url)) return(NA_character_)

  speaker_id <- str_extract(speaker_url, "\\d+")

  mp3_file <- file.path(
    audio_dir,
    paste0(speaker_name, "_", speaker_id, ".mp3")
  )

  mp3_file <- str_replace_all(mp3_file, "[^[:alnum:]_./-]", "_")

  if (!file.exists(mp3_file)) {
    download.file(mp3_url, destfile = mp3_file, mode = "wb", quiet = TRUE)
  }

  mp3_file
}

english_candidates <- pmap_dfr(
  list(
    english_speaker_links$full_url,
    english_speaker_links$speaker_text,
    english_speaker_links$language
  ),
  \(speaker_url, speaker_name, language) {
    message("Checking English candidate: ", speaker_name)

    tryCatch(
      process_speaker_metadata_only(
        speaker_url = speaker_url,
        speaker_name = speaker_name,
        language = language,
        base_url = base_url
      ),
      error = function(e) {
        message("FAILED ENGLISH: ", speaker_name, " | ", conditionMessage(e))

        tibble(
          language = language,
          speaker_name = speaker_name,
          speaker_url = speaker_url,
          birth_place = NA_character_,
          native_language = NA_character_,
          other_languages = NA_character_,
          age_sex = NA_character_,
          age_onset = NA_character_,
          learning_method = NA_character_,
          english_residence = NA_character_,
          length_residence = NA_character_,
          mp3_url = NA_character_
        )
      }
    )
  }
)

english_candidates <- english_candidates %>%
  mutate(
    birth_place_lower = str_to_lower(coalesce(birth_place, "")),
    sex = case_when(
      str_detect(str_to_lower(coalesce(age_sex, "")), "female") ~ "female",
      str_detect(str_to_lower(coalesce(age_sex, "")), "male") ~ "male",
      TRUE ~ NA_character_
    )
  ) %>%
  filter(
    str_detect(birth_place_lower, "usa"),
    sex %in% c("male", "female"),
    !is.na(mp3_url)
  )

set.seed(42)

english_source_speakers <- bind_rows(
  english_candidates %>% filter(sex == "male") %>% slice_sample(n = 1),
  english_candidates %>% filter(sex == "female") %>% slice_sample(n = 1)
) %>%
  mutate(
    mp3_file = pmap_chr(
      list(speaker_name, speaker_url, mp3_url),
      \(speaker_name, speaker_url, mp3_url) {
        download_selected_speaker(
          speaker_name = speaker_name,
          speaker_url = speaker_url,
          mp3_url = mp3_url,
          audio_dir = source_dir
        )
      }
    )
  ) %>%
  select(
    language,
    speaker_name,
    speaker_url,
    birth_place,
    native_language,
    other_languages,
    age_sex,
    sex,
    age_onset,
    learning_method,
    english_residence,
    length_residence,
    mp3_url,
    mp3_file
  )

source_csv_path <- file.path(data_dir, "english_source_speakers.csv")
write_csv(english_source_speakers, source_csv_path)

print(source_csv_path)
print(english_source_speakers)