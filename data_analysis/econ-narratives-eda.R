library(tidyverse)
library(lubridate)
library(readxl)

setwd("~/Documents/econ-indicators/data-analysis")
macro_data = read_delim("data/macro_quant_annotations.csv")

# macro_data %>% 
#   select(spin) %>% 
#   distinct()

macro_data = macro_data %>% 
  mutate(spin = if_else(spin == "pos", "positive", spin),
         spin = if_else(spin == "neg", "negative", spin))

macro_data %>% 
  head()

macro_data %>% 
  select(macro_type) %>% 
  distinct()

# macro_data %>% 
#   filter(macro_type == "prices") %>% 
#   mutate(year = year(date),
#          month = month(date),
#          ym = ym(paste(year, month, sep="-"))) %>% 
#   group_by(source, ym, spin) %>% 
#   summarise(count = n()) %>% 
#   pivot_wider(names_from = spin, values_from = count) %>% 
#   # select(-pos, -neg)  %>% 
#   mutate(total= negative+neutral+positive,
#          per_neg = negative / total,
#          per_neu = neutral / total,
#          per_pos = positive / total) %>% 
#   filter(ym > as.Date("2015-12-01"),
#          source %in% c("nytimes", "wsj")) %>% 
#   ggplot(aes(x=ym, group=source, color=source)) +
#   geom_line(aes(y=per_neg))
  

#
#
# Annual jobs % positive
#
#
macro_data %>% 
  filter(macro_type == "jobs") %>% 
  mutate(year = year(date),
         month = month(date),
         ym = ym(paste(year, month, sep="-"))) %>% 
  group_by(source, year, spin) %>% 
  summarise(count = n()) %>% 
  pivot_wider(names_from = spin, values_from = count) %>% 
  mutate(total= negative+neutral+positive,
         per_neg = negative / total,
         per_neu = neutral / total,
         per_pos = positive / total) %>% 
  filter(year >2015,
         source %in% c("nytimes", "foxnews")) %>% 
  ggplot(aes(x=year, group=source, color=source)) +
  geom_line(aes(y=per_pos)) + 
  geom_vline(aes(xintercept=2016.5), linetype="dashed") + 
  geom_vline(aes(xintercept=2020.5), linetype="dashed") +
  annotate("text", 2017.5, 0.33, label="Trump takes office") + 
  annotate("text", 2021.5, 0.33, label="Biden takes office") + 
  annotate("curve", x=2017.5, y =0.32, xend = 2016.55, yend = 0.285, 
           curvature = -.3, arrow = arrow(length = unit(2, "mm"))) +
  annotate("curve", x=2021.5, y =0.32, xend = 2020.55, yend = 0.285, 
           curvature = -.3, arrow = arrow(length = unit(2, "mm"))) +
  ggtitle("% of 'jobs' quantities rated as positive by year")

# Quarterly
macro_data %>% 
  filter(macro_type == "jobs") %>% 
  mutate(year = year(date),
         month = month(date),
         quarter = as.integer(floor(month / 3)),
         ym = ym(paste(year, month, sep="-")),
         yq = yq(paste(year, quarter, sep="."))) %>% 
  group_by(source, yq, spin) %>% 
  summarise(count = n()) %>% 
  pivot_wider(names_from = spin, values_from = count) %>% 
  mutate(total= negative+neutral+positive,
         per_neg = negative / total,
         per_neu = neutral / total,
         per_pos = positive / total) %>% 
  filter(yq >as.Date("2015-01-01"),
         source %in% c("nytimes", "foxnews")) %>% 
  ungroup() %>% 
  # filter(is.na(yq))
  ggplot(aes(x=yq, group=source, color=source)) +
  geom_line(aes(y=per_pos)) + 
  geom_vline(aes(xintercept=as.Date("2017-01-20")), linetype="dashed") + 
  geom_vline(aes(xintercept=as.Date("2021-01-20")), linetype="dashed") +
  annotate("text", as.Date("2017-11-01"), 0.43, label="Trump takes office") + 
  annotate("text", as.Date("2021-11-01"), 0.33, label="Biden takes office") + 
  annotate("curve", x=as.Date("2017-11-01"), y =0.42, xend = as.Date("2017-02-10"), yend = 0.38, 
           curvature = -.3, arrow = arrow(length = unit(2, "mm"))) +
  annotate("curve", x=as.Date("2021-11-01"), y =0.34, xend = as.Date("2021-02-10"), yend = 0.385, 
           curvature = .3, arrow = arrow(length = unit(2, "mm"))) +
  ggtitle("% of 'jobs' quantities rated as positive by quarter")



#
#
# quantity breakdown
#
#
macro_data %>% 
  mutate(year = year(date),
         month = month(date),
         ym = ym(paste(year, month, sep="-")),
         macro_type = if_else(str_detect(macro_type, "energy|retail|prices"), "prices+", macro_type)) %>% 
  group_by(source, year, macro_type) %>% 
  summarise(count = n()) %>% 
  # pivot_wider(names_from = spin, values_from = count) %>% 
  # mutate(total= negative+neutral+positive,
  #        per_neg = negative / total,
  #        per_neu = neutral / total,
  #        per_pos = positive / total) %>% 
  group_by(source, year) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  filter(year >2015,
         year < 2023,
         source %in% c("nytimes")) %>%
  # head()
  # write_delim("quantity-count-annual-by-source.csv")
  ggplot(aes(x=year, group=macro_type, color=macro_type)) +
  geom_point(aes(y=per)) + 
  geom_line(aes(y=per)) +
  geom_vline(aes(xintercept=2016.5), linetype="dashed") + 
  geom_vline(aes(xintercept=2020.5), linetype="dashed") +
  annotate("text", 2017.5, 0.43, label="Trump takes office") + 
  annotate("text", 2021.5, 0.23, label="Biden takes office") + 
  annotate("curve", x=2017.5, y =0.42, xend = 2016.55, yend = 0.385, 
           curvature = -.3, arrow = arrow(length = unit(2, "mm"))) +
  annotate("curve", x=2021.5, y =0.25, xend = 2020.55, yend = 0.285, 
           curvature = .3, arrow = arrow(length = unit(2, "mm"))) +
  ggtitle("Breakdown of macro quantity types in NYT articles by year (prices, energy, and retail combined)")

macro_data %>% 
  mutate(year = year(date),
         month = month(date),
         ym = ym(paste(year, month, sep="-")),
         macro_type = if_else(str_detect(macro_type, "energy|retail|prices"), "prices+", macro_type)) %>% 
  group_by(source, year, macro_type) %>% 
  summarise(count = n()) %>% 
  group_by(source, year) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  filter(year >2015,
         year < 2023,
         source %in% c("breitbart")) %>% 
  ggplot(aes(x=year, group=macro_type, color=macro_type)) +
  geom_point(aes(y=per)) + 
  geom_line(aes(y=per)) +
  geom_vline(aes(xintercept=2016.5), linetype="dashed") + 
  geom_vline(aes(xintercept=2020.5), linetype="dashed") +
  annotate("text", 2017.5, 0.43, label="Trump takes office") + 
  annotate("text", 2021.5, 0.23, label="Biden takes office") + 
  annotate("curve", x=2017.5, y =0.42, xend = 2016.55, yend = 0.385, 
           curvature = -.3, arrow = arrow(length = unit(2, "mm"))) +
  annotate("curve", x=2021.5, y =0.25, xend = 2020.55, yend = 0.285, 
           curvature = .3, arrow = arrow(length = unit(2, "mm"))) +
  ggtitle("Breakdown of macro quantity types in Breitbart articles by year (prices, energy, and retail combined)")




#
#
# article level
#
#
macro_data %>% 
  mutate(year = year(date),
         month = month(date),
         quarter = as.integer(floor(month / 3)),
         ym = ym(paste(year, month, sep="-")),
         yq = yq(paste(year, quarter, sep="."))) %>% 
  group_by(source, article_id, yq, spin) %>% 
  summarise(count = n()) %>% 
  pivot_wider(names_from = spin, values_from = count) %>% 
  ungroup() %>% 
  mutate(negative = replace_na(negative, 0),
         positive = replace_na(positive, 0),
         neutral = replace_na(neutral, 0),
         total= negative+neutral+positive,
         per_neg = negative / total,
         per_neu = neutral / total,
         per_pos = positive / total,
         article_spin = if_else(per_neg >= 2*per_pos, "negative", "neutral"),
         article_spin = if_else(per_pos >= 2*per_neg, "positive", article_spin)) %>% 
  filter(str_detect(source, "nytimes|foxnews"),
         !is.na(yq)) %>%
  # head()
  # write_delim("article-level-spin-quarterly-by-source.csv")
  group_by(source, yq, article_spin) %>% 
  summarise(count = n()) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  ungroup() %>% 
  filter(yq < as.Date("2023-01-01")) %>% 
  ggplot(aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  geom_line(size=1) +
  ggtitle("Article level spin in Fox and NYT (Quarterly)")


macro_data %>% 
  mutate(year = year(date),
         month = month(date),
         quarter = as.integer(floor(month / 3)),
         ym = ym(paste(year, month, sep="-")),
         yq = yq(paste(year, quarter, sep="."))) %>% 
  group_by(source, article_id, yq, spin) %>% 
  summarise(count = n()) %>% 
  pivot_wider(names_from = spin, values_from = count) %>% 
  ungroup() %>% 
  mutate(negative = replace_na(negative, 0),
         positive = replace_na(positive, 0),
         neutral = replace_na(neutral, 0),
         total= negative+neutral+positive,
         per_neg = negative / total,
         per_neu = neutral / total,
         per_pos = positive / total,
         article_spin = if_else(per_neg >= 2*per_pos, "negative", "neutral"),
         article_spin = if_else(per_pos >= 2*per_neg, "positive", article_spin),
         article_spin = if_else(per_neu >= .5, "neutral", article_spin)) %>% 
  filter(str_detect(source, "nytimes|foxnews"),
         !is.na(yq)) %>% 
  group_by(source, yq, article_spin) %>% 
  summarise(count = n()) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  ungroup() %>% 
  filter(yq < as.Date("2023-01-01")) %>% 
  ggplot(aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  geom_line(size=1) +
  ggtitle("Article level spin in Fox and NYT (Quarterly, Forced Neutral)")





#
#
# LOAD BLS Data
#
#
cpi = read_delim("data/fred-total-cpi.csv", delim = ",")

bls_nonfarm = read_xlsx("data/bls-total-nonfarm-employment.xlsx", skip=12) %>% 
  pivot_longer(!Year, names_to = "Month", values_to = "payroll") %>% 
  mutate(ym = ym(paste(Year, Month, "-")),
         dec_date = Year + ((month(ym)-1) / 12),
         diff  = payroll - lag(payroll))

bls_unemploy = read_xlsx("data/bls-unemployment.xlsx", skip=11) %>% 
  pivot_longer(!Year, names_to = "Month", values_to = "unemployment") %>% 
  mutate(ym = ym(paste(Year, Month, "-")),
         dec_date = Year + ((month(ym)-1) / 12))

bls_nonfarm %>% 
  ggplot(aes(x=ym, y=payroll)) +
  geom_line()

bls_nonfarm %>% 
  mutate(payroll_diff = payroll - lag(payroll)) %>% 
  ggplot(aes(x=ym, y=payroll_diff)) +
  geom_line()


macro_data %>% 
  mutate(year = year(date),
         month = month(date),
         ym = ym(paste(year, month, sep="-")),
         macro_type = if_else(str_detect(macro_type, "energy|retail|prices"), "prices+", macro_type)) %>% 
  group_by(source, year, macro_type) %>% 
  summarise(count = n()) %>% 
  group_by(source, year) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  filter(year >2015,
         source %in% c("nytimes", "foxnews"),
         macro_type == "jobs") %>%
  ggplot() +
  geom_line(data=bls_nonfarm, aes(x = dec_date, y = 0.4 + (payroll - lag(payroll)) / 100000, label="Change in Payroll")) + 
  geom_point(aes(x=year, y=per, group=source, color=source)) + 
  geom_line(aes(x=year, y=per, group=source, color=source)) +
  geom_vline(aes(xintercept=2016.5), linetype="dashed") + 
  geom_vline(aes(xintercept=2020.5), linetype="dashed") +
  xlim(2016, 2023) +
  annotate("text", 2017.6, 0.43, label="Trump takes office") + 
  annotate("text", 2021.6, 0.23, label="Biden takes office") + 
  annotate("curve", x=2017.5, y =0.42, xend = 2016.55, yend = 0.385, 
           curvature = -.3, arrow = arrow(length = unit(2, "mm"))) +
  annotate("curve", x=2021.5, y =0.25, xend = 2020.55, yend = 0.285, 
           curvature = .3, arrow = arrow(length = unit(2, "mm"))) +
  ggtitle("Breakdown of macro quantity types in NYT articles by year (prices, energy, and retail combined)")


#
#
# price+ breakdown 
#
#
macro_data %>% 
  filter(str_detect(macro_type, "energy|retail|prices")) %>% 
  mutate(year = year(date),
         month = month(date),
         quarter = as.integer(floor(month / 3)),
         ym = ym(paste(year, month, sep="-")),
         yq = yq(paste(year, quarter, sep="."))) %>% 
  group_by(source, yq, spin) %>% 
  summarise(count = n()) %>% 
  pivot_wider(names_from = spin, values_from = count) %>% 
  ungroup() %>% 
  mutate(negative = replace_na(negative, 0),
         positive = replace_na(positive, 0),
         neutral = replace_na(neutral, 0),
         total= negative+neutral+positive,
         per_neg = negative / total,
         per_neu = neutral / total,
         per_pos = positive / total) %>% 
  filter(yq >as.Date("2015-01-01"),
         source %in% c("nytimes", "foxnews")) %>% 
  ungroup() %>% 
  # filter(is.na(yq))
  ggplot(aes(x=yq, group=source, color=source)) +
  geom_line(aes(y=per_pos)) + 
  geom_vline(aes(xintercept=as.Date("2017-01-20")), linetype="dashed") + 
  geom_vline(aes(xintercept=as.Date("2021-01-20")), linetype="dashed") +
  annotate("text", as.Date("2017-11-01"), 0.43, label="Trump takes office") + 
  annotate("text", as.Date("2021-11-01"), 0.33, label="Biden takes office") + 
  annotate("curve", x=as.Date("2017-11-01"), y =0.42, xend = as.Date("2017-02-10"), yend = 0.38, 
           curvature = -.3, arrow = arrow(length = unit(2, "mm"))) +
  annotate("curve", x=as.Date("2021-11-01"), y =0.34, xend = as.Date("2021-02-10"), yend = 0.385, 
           curvature = .3, arrow = arrow(length = unit(2, "mm"))) +
  ggtitle("% of 'jobs' quantities rated as positive by quarter")




macro_data %>% 
  filter(str_detect(macro_type, "energy|prices")) %>% 
  mutate(year = year(date),
         month = month(date),
         quarter = as.integer(floor(month / 3)),
         ym = ym(paste(year, month, sep="-")),
         yq = yq(paste(year, quarter, sep="."))) %>% 
  group_by(source, article_id, yq, spin) %>% 
  summarise(count = n()) %>% 
  pivot_wider(names_from = spin, values_from = count) %>% 
  ungroup() %>% 
  mutate(negative = replace_na(negative, 0),
         positive = replace_na(positive, 0),
         neutral = replace_na(neutral, 0),
         total= negative+neutral+positive,
         per_neg = negative / total,
         per_neu = neutral / total,
         per_pos = positive / total,
         article_spin = if_else(per_neg >= 2*per_pos, "negative", "neutral"),
         article_spin = if_else(per_pos >= 2*per_neg, "positive", article_spin)) %>% 
  filter(total > 2) %>%
  filter(str_detect(source, "nytimes"),
         !is.na(yq)) %>%
  # head()
  # write_delim("article-level-spin-quarterly-by-source.csv")
  group_by(source, yq, article_spin) %>% 
  summarise(count = n()) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  ungroup() %>% 
  filter(yq < as.Date("2023-01-01")) %>% 
  ggplot() +
  geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / 8.97)) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2023-01-01")) +
  ggtitle("Article level spin for price/energy articles in the NYT (Quarterly)")



macro_data %>% 
  filter(str_detect(macro_type, "jobs")) %>% 
  mutate(year = year(date),
         month = month(date),
         quarter = as.integer(floor(month / 3)),
         ym = ym(paste(year, month, sep="-")),
         yq = yq(paste(year, quarter, sep="."))) %>% 
  group_by(source, article_id, yq, spin) %>% 
  summarise(count = n()) %>% 
  pivot_wider(names_from = spin, values_from = count) %>% 
  ungroup() %>% 
  mutate(negative = replace_na(negative, 0),
         positive = replace_na(positive, 0),
         neutral = replace_na(neutral, 0),
         total= negative+neutral+positive,
         per_neg = negative / total,
         per_neu = neutral / total,
         per_pos = positive / total,
         article_spin = if_else(per_neg >= 2*per_pos, "negative", "neutral"),
         article_spin = if_else(per_pos >= 2*per_neg, "positive", article_spin)) %>% 
  filter(total > 0) %>%
  filter(str_detect(source, "nytimes"),
         !is.na(yq)) %>%
  # head()
  # write_delim("article-level-spin-quarterly-by-source.csv")
  group_by(source, yq, article_spin) %>% 
  summarise(count = n()) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  ungroup() %>% 
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  geom_line(data=jobs_reporting_per, aes(x=yq, y=per, colour="% of Reporting"), linetype = "dashed", size=1) +
  # geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / 8.97)) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("Change in Payroll" ="#000000", "% of Reporting"="#999999", 
                                "nytimes.negative"="#E69F00", "nytimes.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("Change in Payroll", "% of Reporting", 
                                "% Negative", "% Positive")) +
  # ylim(0,1) + 
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("Article level spin for jobs articles in the NYT (Quarterly)") +
  xlab("Date") +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll")
  ) + 
  theme_bw()


bls_nonfarm %>% 
  mutate(test = (diff) / 5000) %>% view()

jobs_reporting_per = macro_data %>% 
  mutate(year = year(date),
         month = month(date),
         quarter = as.integer(floor(month / 3)),
         ym = ym(paste(year, month, sep="-")),
         yq = yq(paste(year, quarter, sep=".")),
         macro_type = if_else(str_detect(macro_type, "energy|retail|prices"), "prices+", macro_type)) %>% 
  group_by(source, yq, macro_type) %>% 
  summarise(count = n()) %>% 
  group_by(source, yq) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  filter(yq < as.Date("2023-01-01"),
         source %in% c("nytimes"),
         str_detect(macro_type, "jobs"))
