library(tidyverse)
library(readxl)

setwd("~/Documents/econ-indicators/data_analysis")

article_level_annotations = read_delim("data/qual_predictions_edited.csv")


#
#
# BASICS
#
#


article_level_annotations |> 
  head()

article_level_annotations |> 
  group_by(publisher, econ_rate_prediction) |> 
  summarise(count = n()) |> 
  group_by(publisher) |> 
  mutate(total = sum(count),
        per = count / total) |> 
  print(n=50)

article_level_annotations |> 
  group_by(publisher, frame_prediction) |> 
  summarise(count = n()) |> 
  group_by(publisher) |> 
  mutate(total = sum(count),
        per = count / total) |> 
  print(n=50)


article_level_annotations |> 
  filter(str_detect(publisher, "wsj")) |> 
  group_by(publisher, frame_prediction, econ_rate_prediction) |> 
  summarise(count = n()) |> 
  group_by(publisher, frame_prediction) |> 
  mutate(total = sum(count),
        per = count / total) |> 
  print(n=50)

#
#
# ZOOM IN ON MACRO
#
#

article_level_annotations |> 
  filter(str_detect(frame_prediction, "macro")) |> 
  mutate(publisher = as_factor(publisher),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date),
        year = year(date),
        quarter = as.integer(floor(month / 3)),
        ym = ym(paste(year, month, sep="-")),
        yq = yq(paste(year, quarter, sep="."))) |>
  group_by(publisher, month, year, econ_rate_prediction) |> 
  summarise(count = n()) |> 
  group_by(publisher, month, year) |> 
  mutate(total = sum(count),
        per = count / total) |> 
  ungroup() |> 
  mutate(time = year + (month / 12)) |> 
  filter(str_detect(econ_rate_prediction, "good"),
        time >= 2015) |> 
  ggplot(aes(x=time, y=per, group=publisher, color = publisher)) +
  geom_line()

# NYTIMES TEST
article_level_annotations |> 
  filter(str_detect(frame_prediction, "macro")) |> 
  mutate(publisher = as_factor(publisher),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date),
        year = year(date),
        ym = ym(paste(year, month, sep="-")),) |>
  group_by(publisher, ym, econ_rate_prediction) |>
  summarise(count = n()) |> 
  group_by(publisher, ym) |>
  mutate(total = sum(count),
        per = count / total) |> 
  ungroup() |> 
  filter(ym > date("2015-01-01"),
        str_detect(publisher, "nyt"),
        !str_detect(econ_rate_prediction, "irr|none")) |> 
  ggplot(aes(x=ym, y=per, group=econ_rate_prediction, color = econ_rate_prediction)) +
  geom_line() +
  geom_smooth()

# OTHER PUB TEST
article_level_annotations |> 
  filter(str_detect(frame_prediction, "macro")) |> 
  mutate(publisher = as_factor(publisher),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date),
        year = year(date),
        quarter = as.integer(floor(month / 3)),
        yq = yq(paste(year, quarter, sep="."))) |>
  group_by(publisher, yq, econ_rate_prediction) |>
  summarise(count = n()) |> 
  group_by(publisher, yq) |>
  mutate(total = sum(count),
        per = count / total) |> 
  ungroup() |> 
  filter(yq < date("2023-11-01"),
        yq >= date("2015-01-01"),
        str_detect(publisher, "bbc"),
        !str_detect(econ_rate_prediction, "irr|none")) |> 
  ggplot(aes(x=yq, y=per, group=econ_rate_prediction, color = econ_rate_prediction)) +
  geom_line() +
  ggtitle("BBC Macro Article Rating")

article_level_annotations |> 
  filter(str_detect(frame_prediction, "macro")) |> 
  mutate(publisher = as_factor(publisher),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date),
        year = year(date),
        quarter = as.integer(floor(month / 3)),
        yq = yq(paste(year, quarter, sep="."))) |>
  group_by(publisher, yq, econ_rate_prediction) |>
  summarise(count = n()) |> 
  group_by(publisher, yq) |>
  mutate(total = sum(count),
        per = count / total) |> 
  ungroup() |> 
  filter(yq < date("2023-11-01"),
        yq >= date("2015-01-01"),
        !str_detect(econ_rate_prediction, "irr|none|good")) |> 
  ggplot(aes(x=yq, y=per)) +
  geom_line(aes(group=publisher, color = publisher)) +
  ggtitle("Publisher Negativity Levels in Macro Articles") +
  geom_smooth()

article_level_annotations |> 
  filter(str_detect(frame_prediction, "macro")) |> 
  mutate(publisher = as_factor(publisher),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date),
        year = year(date),
        quarter = as.integer(floor(month / 3)),
        yq = yq(paste(year, quarter, sep="."))) |>
  group_by(publisher, yq, econ_rate_prediction) |>
  summarise(count = n()) |> 
  group_by(publisher, yq) |>
  mutate(total = sum(count),
        per = count / total) |> 
  ungroup() |> 
  filter(yq < date("2023-11-01"),
        yq >= date("2015-01-01"),
        !str_detect(econ_rate_prediction, "irr|none|poor")) |> 
  ggplot(aes(x=yq, y=per)) +
  geom_line(aes(group=publisher, color = publisher)) +
  ggtitle("Publisher Positivity Levels in Macro Articles") +
  geom_smooth()


#
#
# SECOND LOOK AT QUANTITIES
#
#

macro_data |> 
  mutate(publisher = as_factor(source),
        month = month(date),
        year = year(date),
        quarter = as.integer(floor(month / 3)),
        yq = yq(paste(year, quarter, sep="."))) |> 
  group_by(publisher, yq, spin) |> 
  summarise(count = n()) |> 
  group_by(publisher, yq) |> 
  mutate(total = sum(count),
        per = count / total) |> 
  filter(str_detect(spin, "negative"),
        yq > date("2015-01-01")) |> 
  ggplot(aes(x=yq, y=per)) +
  geom_line(aes(group=publisher, color=publisher)) 

article_level_annotations |> 
  group_by(publisher, frame_prediction) |> 
  summarise(count = n()) |> 
  group_by(publisher) |> 
  mutate(total = sum(count),
        per = count / total,
        publisher = as_factor(publisher)) |> 
  ggplot(aes(x=publisher, y=per, fill=frame_prediction)) +
  geom_bar(stat="identity", position = "dodge")

joined_data = macro_data |> 
  rename(oid = `...1`) |> 
  rowwise() |> 
  mutate(aid = as.numeric(str_split_1(oid, "_")[1]) ,
        qid = as.numeric(str_split_1(oid, "_")[2]),) |> 
ungroup() |> 
left_join(article_level_annotations, by=c("aid"="article_id", "source"="publisher"))

joined_data |> 
  group_by(source, aid, spin) |> 
  summarise(frame = first(frame_prediction),
            econ_rate_prediction = first(econ_rate_prediction),
            count = n()) |> 
  group_by(source, aid) |> 
  mutate(total = sum(count),
        per = count / total) |> 
  ggplot(aes(x=count, fill=spin)) +
  geom_histogram() +
  xlim(0, 25)

mean_per_by_frame = joined_data |> 
  group_by(source, aid, spin) |> 
  mutate(econ_rate_binary = if_else(str_detect(econ_rate_prediction, "good"), 1, 
                                    if_else(str_detect(econ_rate_prediction, "irr|none"), 2, 0))) |> 
  summarise(frame = first(frame_prediction),
            econ_rate_prediction = first(econ_rate_binary),
            count = n()) |> 
  group_by(source, aid) |> 
  mutate(total = sum(count),
        per = count / total) |> 
  group_by(source, frame, econ_rate_prediction, spin) |> 
  summarise(mean_per = mean(per)) |> 
  filter(str_detect(spin, "negative")) 

mean_per_by_frame |> 
  filter(econ_rate_prediction == 0 | econ_rate_prediction == 2) |> 
  group_by(source) |> 
  summarise(max = max(mean_per),
            min = min(mean_per))

