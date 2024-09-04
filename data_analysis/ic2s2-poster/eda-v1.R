library(tidyverse)
library(ggalluvial)

#
#
# GRAPH 1 - ALL MACRO INDICATORS ARE SPUN IN A NEGATIVE MANNER
#
#

joined_data |>
  mutate(publisher = as_factor(source),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date.x),
        year = year(date.x),
        quarter = as.integer(floor(month / 3)),
        yq = yq(paste(year, quarter, sep=".")),
        ym = ym(paste(year, month, sep="-")),) |>
  filter(!is.na(macro_type)) |>
  group_by(ym, macro_type, spin) |>
  summarise(count = n()) |> 
  group_by(ym, macro_type) |> 
  mutate(total = sum(count),
        per = count / total) |>
  filter(str_detect(spin, "neg")) |> 
  ggplot(aes(x=ym, y=per)) +
  geom_line(aes(group = macro_type, color = macro_type), alpha=0.5) +
  # geom_smooth() +
  xlim(date("2015-01-01"), date("2023-10-01"))
  

joined_data |>
  # filter(str_detect(source, "nytime")) |> 
  mutate(publisher = as_factor(source),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date.x),
        year = year(date.x),
        quarter = as.integer(floor(month / 3)),
        yq = yq(paste(year, quarter, sep=".")),
        ym = ym(paste(year, month, sep="-")),) |>
  filter(!is.na(macro_type)) |>
  group_by(ym, spin) |>
  summarise(count = n()) |> 
  group_by(ym) |> 
  mutate(total = sum(count),
        per = count / total) |>
  ggplot(aes(x=ym, y=per, group = spin, color = spin)) +
  geom_line() +
  xlim(date("2015-01-01"), date("2023-11-01")) +
  ggtitle("% of Quantities Per Class")

joined_data |>
  mutate(publisher = as_factor(source),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date.x),
        year = year(date.x),
        quarter = as.integer(floor(month / 2)),
        yq = yq(paste(year, quarter, sep=".")),
        ym = ym(paste(year, month, sep="-")),) |>
  # filter(!is.na(macro_type)) |>
  filter(str_detect(frame_prediction, "macro")) |> 
  group_by(yq, econ_rate_prediction) |>
  summarise(count = n()) |> 
  group_by(yq) |> 
  mutate(total = sum(count),
        per = count / total) |>
  filter(!str_detect(econ_rate_prediction, "irre|NA")) |> 
  ggplot(aes(x=yq, y=per, group = econ_rate_prediction, color = econ_rate_prediction)) +
  geom_line() +
  xlim(date("2015-01-01"), date("2023-11-01"))


#
#
# GRAPH 2 - HOW DOES THE MEAN SPIN SHIFT OVER TIME
#
#

joined_data |>
  mutate(publisher = as_factor(source),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date.x),
        year = year(date.x),
        quarter = as.integer(floor(month / 3)),
        yq = yq(paste(year, quarter, sep=".")),
        ym = ym(paste(year, month, sep="-")),
        spin_level = if_else(str_detect(spin, "pos"), 1,
                        if_else(str_detect(spin, "neu"), 0.5, 0))) |>
  filter(!is.na(macro_type)) |>
  group_by(ym) |>
  summarise(mean_spin = mean(spin_level)) |> 
  ggplot(aes(x=ym, y=mean_spin)) +
  geom_line() +
  xlim(date("2015-01-01"), date("2023-11-01"))

joined_data |>
  filter(!str_detect(econ_rate_prediction, "irre|NA")) |> 
  mutate(publisher = as_factor(source),
        # econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date.x),
        year = year(date.x),
        quarter = as.integer(floor(month / 2)),
        yq = yq(paste(year, quarter, sep=".")),
        ym = ym(paste(year, month, sep="-")),
        rate_level = if_else(str_detect(econ_rate_prediction, "good"), 1,
                        if_else(str_detect(econ_rate_prediction, "none"), 0.5, 0))) |>
  # filter(!is.na(macro_type)) |>
  # filter(str_detect(frame_prediction, "macro")) |> 
  group_by(yq) |>
  summarise(mean_level = mean(rate_level)) |> 
  ggplot(aes(x=yq, y=mean_level)) +
  geom_line() +
  xlim(date("2015-01-01"), date("2023-11-01"))

#
#
# GRAPH 3 - HOW DOES THE USE OF INDICATORS SHIFT OVER TIME (ALLUVIAL?)
#
#

macro_type_per_data = joined_data |>
  mutate(publisher = as_factor(source),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date.x),
        year = year(date.x),
        quarter = as.double(floor(month / 3)),
        # yq = yq(paste(year, quarter, sep=".")),
        yq = as.double(year) + as.double((quarter - 1) / 4),
        ym = ym(paste(year, month, sep="-")),
        macro_type = if_else(str_detect(macro_type, "energy|prices"), "prices+", macro_type),
        macro_type = if_else(str_detect(macro_type, "currency|market"), "markets+", macro_type),
        macro_type = if_else(str_detect(macro_type, "macro|interest|retail"), "macro+", macro_type),) |>
  filter(!is.na(macro_type)) |>
  group_by(yq, publisher, macro_type) |>
  summarise(count = n()) |> 
  group_by(yq, publisher) |> 
  mutate(total = sum(count),
        per = count / total) |>
  ungroup() |> 
  mutate(macro_type = as.factor(macro_type)) 

# macro_type_per_data$yq[1]
macro_type_per_data |>
  # select(yq) |> 
  # head()
  filter(str_detect(publisher, "nyt")) |> 
  select(yq, macro_type, per) |> 
  # mutate(year = as.numeric(year)) |> 
  filter(yq >= 2021, yq <= 2022.8) |> 
  ggplot(aes(x=yq, y=per, color=macro_type))+
  geom_line()

  ggplot(aes(x=yq, y=count, alluvium=macro_type, stratum=macro_type)) +
  # geom_flow()
  geom_alluvium(aes(fill = macro_type, colour = macro_type),
                alpha = .75,
                decreasing = NA, width = 0, knot.pos = 0) +
  # geom_stratum(alpha = 0.5) +
  # geom_text(aes(label = macro_type), stat = "stratum", size = 2) +
  theme(legend.position = "none")

macro_type_per_data

joined_data |>
  filter(str_detect(source, "nytime")) |> 
  mutate(publisher = as_factor(source),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date.x),
        year = year(date.x),
        quarter = as.integer(floor(month / 3)),
        yq = yq(paste(year, quarter, sep=".")),
        ym = ym(paste(year, month, sep="-")),
        macro_type = if_else(str_detect(macro_type, "energy|prices"), "prices+", macro_type),
        macro_type = if_else(str_detect(macro_type, "currency|market"), "markets+", macro_type),
        macro_type = if_else(str_detect(macro_type, "macro|interest|retail"), "macro+", macro_type),) |>
  filter(!is.na(macro_type)) |>
  group_by(yq, macro_type) |>
  summarise(count = n()) |> 
  group_by(yq) |> 
  mutate(total = sum(count),
        per = count / total) |>
  ungroup() |> 
  mutate(macro_type = as.factor(macro_type)) |> 
  ggplot(aes(x=yq, y=per, group = spin, color = spin)) +
  geom_line() +
  xlim(date("2015-01-01"), date("2023-11-01"))

#
#
# GRAPH 4 - HOW GREAT IS THE BIAS TOWARDS NEGATIVITY
#
#

#
#
# GRAPH 5 -  WHAT DOES THE VOLUME OF ECONOMIC REPORTING LOOK LIKE (# NEG VS # POS)
#
#

joined_data |>
  mutate(publisher = as_factor(source),
        econ_rate_prediction = as_factor(econ_rate_prediction),
        econ_change_prediction = as_factor(econ_change_prediction),
        month = month(date.x),
        year = year(date.x),
        quarter = as.integer(floor(month / 2)),
        yq = yq(paste(year, quarter, sep=".")),
        ym = ym(paste(year, month, sep="-")),) |>
  # filter(!is.na(macro_type)) |>
  filter(str_detect(frame_prediction, "macro")) |> 
  group_by(ym, econ_rate_prediction) |>
  summarise(count = n()) |> 
  group_by(ym) |> 
  mutate(total = sum(count),
        per = count / total) |>
  filter(!str_detect(econ_rate_prediction, "irre|NA")) |> 
  ggplot(aes(x=ym, y=count, group = econ_rate_prediction, color = econ_rate_prediction)) +
  geom_line() +
  xlim(date("2015-01-01"), date("2023-11-01"))


#
#
# GRAPH 6 - CAN WE SHOW A POLITICAL ANGLE E.G. FOCUS ON 2022 MIDTERM OR 2016 ELECTION
#
#

joined_data |> 
  mutate(rate_level = if_else(str_detect(econ_rate_prediction, "good"), 1,
                      if_else(str_detect(econ_rate_prediction, "none"), 0.5, 0)),
        period = if_else(date.x < as_date("2017-02-01"), 0, 
                  if_else(date.x >= as_date("2017-02-01") & 
                          date.x < as_date("2020-03-01"), 1, 
                  if_else(date.x >= as_date("2020-03-01") & 
                          date.x < as_date("2021-02-01"), 2, 3))),
        period = as.factor(period)) |> 
  group_by(source, period, date.x) |> 
  filter(!is.na(rate_level)) |> 
  summarise(mean_rate = mean(rate_level)) |> 
  group_by(source, period) |> 
  summarise(mean_level = mean(mean_rate)) |> 
  print(n=50)
  ggplot(aes(x=mean_rate, group=period, fill=period)) +
  geom_density(alpha=0.5)

joined_data |> 
  mutate(rate_level = if_else(str_detect(econ_rate_prediction, "good"), 1,
                      if_else(str_detect(econ_rate_prediction, "none"), 0.5, 0)),
        period = if_else(date.x < as_date("2017-02-01"), 0, 
                  if_else(date.x >= as_date("2017-02-01") & 
                          date.x < as_date("2020-03-01"), 1, 
                  if_else(date.x >= as_date("2020-03-01") & 
                          date.x < as_date("2021-02-01"), 2, 3))),
        period = as.factor(period)) |> 
  group_by(source, period, date.x) |> 
  filter(!is.na(rate_level)) |> 
  summarise(mean_rate = mean(rate_level)) |> 
  group_by(source, period) |> 
  summarise(mean_level = mean(mean_rate)) |> 
  print(n=50)

joined_data |> 
  mutate(rate_level = if_else(str_detect(econ_rate_prediction, "good"), 1,
                      if_else(str_detect(econ_rate_prediction, "none"), 0.5, 0)),
        period = if_else(date.x < as_date("2017-02-01"), 0, 
                  if_else(date.x >= as_date("2017-02-01") & 
                          date.x < as_date("2020-03-01"), 1, 
                  if_else(date.x >= as_date("2020-03-01") & 
                          date.x < as_date("2021-02-01"), 2, 0))),
        period = as.factor(period)) |> 
  group_by(source, period, date.x) |> 
  filter(!is.na(rate_level)) |> 
  summarise(mean_rate = mean(rate_level)) |> 
  # group_by(source, period) |> 
  filter(period == 0) |> 
  head()
  group_by(source) |> 
    # group_by(Cell_line, Gene, ID) %>%
  group_map(~ t.test(mean_rate ~ source))
  
  summarise(mean_level = mean(mean_rate)) |> 
  print(n=50)

#
#
# Frame prevalence
#
#


joined_data |> 
  filter(str_detect(source, "nyt")) |> 
  group_by(aid) |> 
  summarise(date = first(date.x),
            frame_prediction = first(frame_prediction),
            econ_rate_prediction = first(econ_rate_prediction),
            econ_change_prediction = first(econ_change_prediction)) |> 
  filter(!str_detect(econ_rate_prediction, "irr|NA")) |> 
  group_by(date, econ_rate_prediction) |> 
  summarise(count = n()) |> 
  ggplot(aes(x=date, y=count, color = econ_rate_prediction)) +
  geom_line(alpha=0.5)

joined_data |> 
  filter(str_detect(source, "nyt")) |> 
  group_by(aid) |> 
  summarise(date = first(date.x),
            frame_prediction = first(frame_prediction),
            econ_rate_prediction = first(econ_rate_prediction),
            econ_change_prediction = first(econ_change_prediction),
            month = month(date.x),
            year = year(date.x),
            quarter = as.integer(floor(month / 2)),
            yq = yq(paste(year, quarter, sep=".")),
            ym = ym(paste(year, month, sep="-")),) |> 
  # filter(!str_detect(econ_rate_prediction, "irr|NA")) |> 
  group_by(ym, frame_prediction) |> 
  summarise(count = n()) |> 
  ggplot(aes(x=ym, y=count, color = frame_prediction)) +
  geom_line(alpha=0.5)


#
#
# MACRO SUMMARY
#
#

joined_data |> 
  # filter(str_detect(source, "nyt")) |> 
  filter(str_detect(frame_prediction, "macro")) |> 
  group_by(date.x, aid) |> 
  summarise(count = n()) |> 
  arrange(date.x) |> 
  group_by(date.x) |> 
  summarise(mean_count = mean(count)) |> 
  ungroup() |> 
  mutate(lag1 = lag(mean_count, 1),
        lag2 = lag(mean_count, 2),
        lag3 = lag(mean_count, 3),
        lag4 = lag(mean_count, 4),
        lag5 = lag(mean_count, 5),
        lag6 = lag(mean_count, 6),
        lag7 = lag(mean_count, 7),
        rolling_mean = (lag1 + lag2 + lag3 + lag4 + lag5 +lag6 +lag7) / 7,
        period = if_else(date.x >= as_date("2015-01-01") &
                        date.x < as_date("2017-02-01"), 0, 
                if_else(date.x >= as_date("2017-02-01") & 
                        date.x < as_date("2020-03-01"), 1, 
                if_else(date.x >= as_date("2020-03-01") & 
                        date.x < as_date("2021-02-01"), 2, 3))),
        period = as.factor(period)) |> 
  group_by(period) |> 
  mutate(mean = mean(mean_count)) |> 
  filter(date.x >= as_date("2015-01-01"), date.x < as_date("2022-11-01")) |> 
  ggplot(aes(x=date.x, y=rolling_mean)) +
  geom_line() +
  geom_line(aes(x=date.x, y=mean, group=period, color = period), size = 1.5) +
  ggtitle("Indicators Per Article")


joined_data |> 
  filter(str_detect(frame_prediction, "macro")) |> 
  # filter(str_detect(source, "nyt|wsj|wash")) |> 
  # filter(str_detect(source, "fox|breit")) |> 
  group_by(aid) |> 
  slice_head(n=1) |> 
  group_by(date.x) |> 
  summarise(count = n()) |> 
  arrange(date.x) |> 
  ungroup() |> 
  mutate(lag1 = lag(count, 1),
        lag2 = lag(count, 2),
        lag3 = lag(count, 3),
        lag4 = lag(count, 4),
        lag5 = lag(count, 5),
        lag6 = lag(count, 6),
        lag7 = lag(count, 7),
        rolling_mean = (lag1 + lag2 + lag3 + lag4 + lag5 +lag6 +lag7) / 7,
        period = if_else(date.x >= as_date("2015-01-01") &
                          date.x < as_date("2017-02-01"), 0, 
                  if_else(date.x >= as_date("2017-02-01") & 
                          date.x < as_date("2020-03-01"), 1, 
                  if_else(date.x >= as_date("2020-03-01") & 
                          date.x < as_date("2021-02-01"), 2, 3))),
        period = as.factor(period)) |> 
  group_by(period) |> 
  mutate(mean = mean(count)) |> 
  filter(date.x >= as_date("2015-01-01"), date.x < as_date("2022-11-01")) |> 
  ggplot(aes(x=date.x, y=rolling_mean)) +
  geom_line() +
  # geom_smooth(aes(group=period, color=period), method = "lm")
  geom_line(aes(x=date.x, y=mean, group=period, color = period), size = 1.5) +
  ggtitle("Articles Per Day")

#
#
# MACRO NEGATIVITY
#
#


joined_data |> 
  filter(str_detect(frame_prediction, "macro")) |> 
  # filter(str_detect(source, "huff|nyt|wash|bbc")) |> 
    filter(str_detect(source, "nyt|wash")) |>
  # filter(str_detect(source, "wsj")) |> 
  # filter(str_detect(source, "fox|breit")) |> 
  # filter(str_detect(source, "huff")) |> 
  group_by(aid) |> 
  slice_head(n=1) |> 
  ungroup() |> 
  mutate(rate_level = if_else(str_detect(econ_rate_prediction, "good"), 1,
                      if_else(str_detect(econ_rate_prediction, "none"), 0.5, 0))) |> 
  group_by(date.x) |> 
  summarise(mean_rate = mean(rate_level)) |> 
  arrange(date.x) |> 
  ungroup() |> 
  mutate(lag1 = lag(mean_rate, 1),
        lag2 = lag(mean_rate, 2),
        lag3 = lag(mean_rate, 3),
        lag4 = lag(mean_rate, 4),
        lag5 = lag(mean_rate, 5),
        lag6 = lag(mean_rate, 6),
        lag7 = lag(mean_rate, 7),
        rolling_mean = (lag1 + lag2 + lag3 + lag4 + lag5 +lag6 +lag7) / 7,
        period = if_else(date.x >= as_date("2015-01-01") &
                          date.x < as_date("2017-02-01"), 0, 
                  if_else(date.x >= as_date("2017-02-01") & 
                          date.x < as_date("2020-03-01"), 1, 
                  if_else(date.x >= as_date("2020-03-01") & 
                          date.x < as_date("2021-02-01"), 2, 3))),
        period = as.factor(period)) |> 
  group_by(period) |> 
  mutate(mean = mean(mean_rate)) |> 
  filter(date.x >= as_date("2015-01-01"), date.x < as_date("2022-11-01")) |> 
  ggplot(aes(x=date.x, y=rolling_mean)) +
  geom_line() +
  # geom_smooth(aes(group=period, color=period), method = "lm")
  geom_line(aes(x=date.x, y=mean, group=period, color = period), size = 1.5) +
  ggtitle("Mean Rating (NYT + WAPO)")


#
#
# FRAME POPULARITY
#
#


joined_data |> 
  # filter(str_detect(source, "nyt|wsj|wash")) |> 
  # filter(str_detect(source, "fox|breit")) |> 
  group_by(aid) |> 
  slice_head(n=1) |> 
  group_by(date.x, frame_prediction) |> 
  summarise(count = n()) |> 
  arrange(date.x) |> 
  # ungroup() |> 
  group_by(frame_prediction) |> 
  mutate(lag1 = lag(count, 1),
        lag2 = lag(count, 2),
        lag3 = lag(count, 3),
        lag4 = lag(count, 4),
        lag5 = lag(count, 5),
        lag6 = lag(count, 6),
        lag7 = lag(count, 7),
        rolling_mean = (lag1 + lag2 + lag3 + lag4 + lag5 +lag6 +lag7) / 7,
        period = if_else(date.x >= as_date("2015-01-01") &
                          date.x < as_date("2017-02-01"), 0, 
                  if_else(date.x >= as_date("2017-02-01") & 
                          date.x < as_date("2020-03-01"), 1, 
                  if_else(date.x >= as_date("2020-03-01") & 
                          date.x < as_date("2021-02-01"), 2, 3))),
        period = as.factor(period)) |> 
  # group_by(period) |> 
  # mutate(mean = mean(count)) |> 
  filter(date.x >= as_date("2015-01-01"), date.x < as_date("2022-11-01")) |> 
  ggplot(aes(x=date.x, y=rolling_mean)) +
  geom_line(aes(color = frame_prediction)) +
  # geom_smooth(aes(group=period, color=period), method = "lm")
  # geom_line(aes(x=date.x, y=mean, group=period, color = period), size = 1.5) +
  ggtitle("Framed Articles Per Day")
