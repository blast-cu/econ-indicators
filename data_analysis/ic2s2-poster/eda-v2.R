cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

#
#
# Negativity bias
#
#

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
  mutate(ym = as.Date(ym)) |> 
  ggplot(aes(x=ym, y=per, group = spin, color = spin)) +
  geom_line() +
  scale_x_date(limits=c(as.Date("2015-01-01"), as.Date("2022-10-30"))) +
  ggtitle("Economic Indicators Are Often Negative") +
  theme_bw() +
  theme(legend.position = "bottom") +
  xlab("Date") +
  ylab("Proportion (0-1)") +
  scale_color_manual(values = c("negative"="#CC79A7", 
                                "neutral"="#E69F00", "positive"="#56B4E9"),
                              name="",
                              labels = c("Negative", "Neutral", 
                                "Positive"))

#
#
# Indicators per day
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
        Period = if_else(date.x >= as_date("2015-01-01") &
                        date.x < as_date("2017-02-01"), 0, 
                if_else(date.x >= as_date("2017-02-01") & 
                        date.x < as_date("2020-03-01"), 1, 
                if_else(date.x >= as_date("2020-03-01") & 
                        date.x < as_date("2021-02-01"), 2, 3))),
        Period = as.factor(Period)) |> 
  group_by(Period) |> 
  mutate(mean = mean(mean_count)) |> 
  filter(date.x >= as_date("2015-01-01"), date.x < as_date("2022-11-01")) |> 
  ggplot(aes(x=date.x, y=rolling_mean)) +
  geom_line() +
  geom_line(aes(x=date.x, y=mean, group=Period, color = Period), size = 1.5) +
  ggtitle("Indicators Per Macroeconomic Article") +
  theme_bw() +
  theme(legend.position = "bottom") +
  xlab("Date") +
  ylab("Economic Quantity Count") +
  scale_color_manual(values = c("0" ="#009E73", "1"="#CC79A7", 
                                "2"="#E69F00", "3"="#56B4E9"),
                              name="",
                              labels = c("Obama", "Trump (Pre-Covid)", 
                                "Trump (Covid)", "Biden"))
  


#
#
# Articles per day
#
#

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
        Period = if_else(date.x >= as_date("2015-01-01") &
                          date.x < as_date("2017-02-01"), 0, 
                  if_else(date.x >= as_date("2017-02-01") & 
                          date.x < as_date("2020-03-01"), 1, 
                  if_else(date.x >= as_date("2020-03-01") & 
                          date.x < as_date("2021-02-01"), 2, 3))),
        Period = as.factor(Period)) |> 
  group_by(Period) |> 
  mutate(mean = mean(count)) |> 
  filter(date.x >= as_date("2015-01-01"), date.x < as_date("2022-11-01")) |> 
  ggplot(aes(x=date.x, y=rolling_mean)) +
  geom_line() +
  # geom_smooth(aes(group=Period, color=Period), method = "lm")
  geom_line(aes(x=date.x, y=mean, group=Period, color = Period), size = 1.5) +
  ggtitle("Macroeconomic Articles Collected") +
  theme_bw() +
  theme(legend.position = "bottom") +
  xlab("Date") +
  ylab("Article Count") +
  scale_color_manual(values = c("0" ="#009E73", "1"="#CC79A7", 
                                "2"="#E69F00", "3"="#56B4E9"),
                              name="",
                              labels = c("Obama", "Trump (Pre-Covid)", 
                                "Trump (Covid)", "Biden"))

#
#
# Political Angle
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
        Period = if_else(date.x >= as_date("2015-01-01") &
                          date.x < as_date("2017-02-01"), 0, 
                  if_else(date.x >= as_date("2017-02-01") & 
                          date.x < as_date("2020-03-01"), 1, 
                  if_else(date.x >= as_date("2020-03-01") & 
                          date.x < as_date("2021-02-01"), 2, 3))),
        Period = as.factor(Period)) |> 
  group_by(Period) |> 
  mutate(mean = mean(mean_rate)) |> 
  filter(date.x >= as_date("2015-01-01"), date.x < as_date("2022-11-01")) |> 
  ggplot(aes(x=date.x, y=rolling_mean)) +
  geom_line() +
  # geom_smooth(aes(group=Period, color=Period), method = "lm")
  geom_line(aes(x=date.x, y=mean, group=Period, color = Period), size = 1.5) +
  ggtitle("Mean Economic Rating (NYT + WAPO)") +
  theme_bw() +
  theme(legend.position = "bottom") +
  # theme(axis.title.y = element_blank()) +
  xlab("Date") +
  ylab("Rating (Lower Values Are More Negative)") +
  scale_color_manual(values = c("0" ="#009E73", "1"="#CC79A7", 
                                "2"="#E69F00", "3"="#56B4E9"),
                              name="",
                              labels = c("Obama", "Trump (Pre-Covid)", 
                                "Trump (Covid)", "Biden"))

joined_data |> 
  filter(str_detect(frame_prediction, "macro")) |> 
  # filter(str_detect(source, "wsj")) |> 
  filter(str_detect(source, "fox|breit")) |> 
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
        Period = if_else(date.x >= as_date("2015-01-01") &
                          date.x < as_date("2017-02-01"), 0, 
                  if_else(date.x >= as_date("2017-02-01") & 
                          date.x < as_date("2020-03-01"), 1, 
                  if_else(date.x >= as_date("2020-03-01") & 
                          date.x < as_date("2021-02-01"), 2, 3))),
        Period = as.factor(Period)) |> 
  group_by(Period) |> 
  mutate(mean = mean(mean_rate)) |> 
  filter(date.x >= as_date("2015-01-01"), date.x < as_date("2022-11-01")) |> 
  ggplot(aes(x=date.x, y=rolling_mean)) +
  geom_line() +
  # geom_smooth(aes(group=Period, color=Period), method = "lm")
  geom_line(aes(x=date.x, y=mean, group=Period, color = Period), size = 1.5) +
  ggtitle("Mean Economic Rating (Fox + Breitbart)") +
  theme_bw() +
  theme(legend.position = "bottom") +
  theme(axis.title.y = element_blank?()) +
  xlab("Date") +
  ylab("Rating (Lower Values Are More Negative)") +
  scale_color_manual(values = c("0" ="#009E73", "1"="#CC79A7", 
                                "2"="#E69F00", "3"="#56B4E9"),
                              name="",
                              labels = c("Obama", "Trump (Pre-Covid)", 
                                "Trump (Covid)", "Biden"))

joined_data |> 
  filter(str_detect(frame_prediction, "macro")) |> 
  filter(str_detect(source, "wsj")) |>  
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
        Period = if_else(date.x >= as_date("2015-01-01") &
                          date.x < as_date("2017-02-01"), 0, 
                  if_else(date.x >= as_date("2017-02-01") & 
                          date.x < as_date("2020-03-01"), 1, 
                  if_else(date.x >= as_date("2020-03-01") & 
                          date.x < as_date("2021-02-01"), 2, 3))),
        Period = as.factor(Period)) |> 
  group_by(Period) |> 
  mutate(mean = mean(mean_rate)) |> 
  filter(date.x >= as_date("2015-01-01"), date.x < as_date("2022-11-01")) |> 
  ggplot(aes(x=date.x, y=rolling_mean)) +
  geom_line() +
  # geom_smooth(aes(group=Period, color=Period), method = "lm")
  geom_line(aes(x=date.x, y=mean, group=Period, color = Period), size = 1.5) +
  ggtitle("Mean Economic Rating (WSJ)") +
  theme_bw() +
  theme(legend.position = "bottom") +
    # theme(axis.title.y = element_blank()) +
  xlab("Date") +
  ylab("Rating (Lower Values Are More Negative)") +
  scale_color_manual(values = c("0" ="#009E73", "1"="#CC79A7", 
                                "2"="#E69F00", "3"="#56B4E9"),
                              name="",
                              labels = c("Obama", "Trump (Pre-Covid)", 
                                "Trump (Covid)", "Biden"))




# joined_data |> 
#   filter(str_detect(frame_prediction, "macro")) |> 
#   group_by(aid) |> 
#   slice_head(n=1) |> 
#   ungroup() |> 
#   mutate(rate_level = if_else(str_detect(econ_rate_prediction, "good"), 1,
#                       if_else(str_detect(econ_rate_prediction, "none"), 0.5, 0)),
#         Lean = if_else(str_detect(source, "wsj"), "WSJ",
#                if_else(str_detect(source, "nyt|wash"), "NYT + WAPO",
#                if_else(str_detect(source, "fox|breit"), "Fox + Breitbart", "Other")))) |>
#   group_by(Lean, date.x) |> 
#   summarise(mean_rate = mean(rate_level)) |> 
#   arrange(date.x) |> 
#   ungroup() |> 
#   mutate(lag1 = lag(mean_rate, 1),
#         lag2 = lag(mean_rate, 2),
#         lag3 = lag(mean_rate, 3),
#         lag4 = lag(mean_rate, 4),
#         lag5 = lag(mean_rate, 5),
#         lag6 = lag(mean_rate, 6),
#         lag7 = lag(mean_rate, 7),
#         rolling_mean = (lag1 + lag2 + lag3 + lag4 + lag5 +lag6 +lag7) / 7,
#         Period = if_else(date.x >= as_date("2015-01-01") &
#                           date.x < as_date("2017-02-01"), 0, 
#                   if_else(date.x >= as_date("2017-02-01") & 
#                           date.x < as_date("2020-03-01"), 1, 
#                   if_else(date.x >= as_date("2020-03-01") & 
#                           date.x < as_date("2021-02-01"), 2, 3))),
#         Period = as.factor(Period),) |> 
#   group_by(Lean, Period) |> 
#   mutate(mean = mean(mean_rate)) |> 
#   filter(date.x >= as_date("2015-01-01"), date.x < as_date("2022-11-01")) |> 
#   ggplot(aes(x=date.x, y=rolling_mean)) +
#   geom_line() +
#   # geom_smooth(aes(group=Period, color=Period), method = "lm")
#   geom_line(aes(x=date.x, y=mean, group=Period, color = Period), size = 1.5) +
#   facet_wrap(~Lean, ncol=1) +
#   ggtitle("Mean Economic Rating (WSJ)") +
#   theme_bw() +
#   theme(legend.position = "bottom") +
#   xlab("Date") +
#   ylab("Rating (Lower Values Are More Negative)") +
#   scale_color_manual(values = c("0" ="#009E73", "1"="#CC79A7", 
#                                 "2"="#E69F00", "3"="#56B4E9"),
#                               name="",
#                               labels = c("Obama", "Trump (Pre-Covid)", 
#                                 "Trump (Covid)", "Biden"))
                                