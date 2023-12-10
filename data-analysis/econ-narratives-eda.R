library(tidyverse)
library(lubridate)

macro_data = read_delim("~/Documents/macro_quant_annotations.csv")

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

macro_data %>% 
  filter(macro_type == "prices") %>% 
  mutate(year = year(date),
         month = month(date),
         ym = ym(paste(year, month, sep="-"))) %>% 
  group_by(source, ym, spin) %>% 
  summarise(count = n()) %>% 
  pivot_wider(names_from = spin, values_from = count) %>% 
  # select(-pos, -neg)  %>% 
  mutate(total= negative+neutral+positive,
         per_neg = negative / total,
         per_neu = neutral / total,
         per_pos = positive / total) %>% 
  filter(ym > as.Date("2015-12-01"),
         source %in% c("nytimes", "wsj")) %>% 
  ggplot(aes(x=ym, group=source, color=source)) +
  geom_line(aes(y=per_neg))
  

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



macro_data %>% 
  mutate(year = year(date),
         month = month(date),
         ym = ym(paste(year, month, sep="-"))) %>% 
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
         source %in% c("nytimes")) %>% 
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
  ggtitle("Breakdown of macro quantity types in NYT articles by year")

