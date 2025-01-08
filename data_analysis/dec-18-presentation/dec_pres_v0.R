library(jsonlite)
library(tidyverse)
library(ggthemes)

setwd("~/Documents/econ-indicators/data_analysis/dec-18-presentation")

#
# load data
#

temp_json = fromJSON("../data/dec_3_24_data/articles.json", simplifyVector = TRUE)

article_data = tibble(id = names(temp_json),
       data = map(temp_json, as.data.frame)) %>%
  unnest_wider(data) %>%
  unnest(cols = names(.)) %>% 
  mutate(frame = as_factor(frame),
         econ_change = as_factor(econ_change),
         econ_rate = as_factor(econ_rate),
         source = as_factor(source),
         date = as_date(date),
         id = as.numeric(id))

temp_json = fromJSON("../data/dec_3_24_data/quants.json", simplifyVector = TRUE)

quant_data = tibble(data = temp_json) %>%
  unnest_wider(data) %>%
  unnest(cols = names(.)) %>% 
  mutate(type = as_factor(type),
         macro_type = as_factor(macro_type),
         spin = as_factor(spin),
         article_id = as.numeric(article_id))

print("Data loaded.")
rm(temp_json)

#
# Quick test
# 

# Articles

article_data %>% 
  filter(str_detect(source, "nytimes"),
         str_detect(frame, "macro")) %>% 
  mutate(year = year(date),
         month = month(date)) %>% 
  group_by(source, year, month, econ_rate) %>% 
  summarise(count = n()) %>% 
  group_by(source, year, month) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, month, "1", sep="-"))) %>% 
  ungroup() %>%
  ggplot(aes(x=date, y=per, group=econ_rate)) +
  geom_line(aes(color=econ_rate))

article_data %>% 
  filter(str_detect(frame, "macro"),
         str_detect(source, "nyt|wash")) %>% 
  mutate(year = year(date),
         month = month(date)) %>% 
  group_by(source, year, month, econ_rate) %>% 
  summarise(count = n()) %>% 
  group_by(source, year, month) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, month, "1", sep="-"))) %>% 
  ungroup() %>%
  filter(str_detect(econ_rate, "poor"),
         year > 2014) %>% 
  ggplot(aes(x=date, y=per)) +
  geom_line(aes(color=source, group=source)) +
  geom_smooth(se=FALSE) +
  ggtitle("% of Macroeconomic Articles That Were Negative (Per Month)") +
  xlab("Date") +
  ylab("Proportion")

# quants

quant_data %>% 
  left_join(article_data, by=c("article_id"="id")) %>% 
  filter(str_detect(frame, "macro"),
         str_detect(type, "macro"),
         str_detect(source, "nytimes|washington|wsj")) %>% 
  mutate(year = year(date),
         month = month(date)) %>% 
  group_by(source, year, month, macro_type) %>% 
  summarise(count = n()) %>% 
  group_by(source, year, month) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, month, "1", sep="-"))) %>% 
  ungroup() %>%
  filter(year > 2014) %>% 
  head()

quant_data %>% 
  left_join(article_data, by=c("article_id"="id")) %>% 
  filter(str_detect(frame, "macro"),
         str_detect(type, "macro"),
         str_detect(source, "nyt")) %>% 
  mutate(year = year(date),
         month = month(date)) %>% 
  group_by(year, month, macro_type) %>% 
  summarise(count = n()) %>% 
  group_by(year, month) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, month, "1", sep="-"))) %>% 
  ungroup() %>%
  filter(year > 2014,
         str_detect(macro_type,"job|prices|macro")) %>% 
  ggplot(aes(x=date, y=per)) +
  geom_line(aes(color=macro_type, group=macro_type), size=1, alpha=0.7) +
  ggtitle("Macro Indicator Use (NYT)")

quant_data %>% 
  left_join(article_data, by=c("article_id"="id")) %>% 
  filter(str_detect(frame, "macro"),
         str_detect(type, "macro"),
         str_detect(source, "wsj|nyt|wash")) %>% 
  mutate(year = year(date),
         month = month(date)) %>% 
  group_by(source, year, month, macro_type) %>% 
  summarise(count = n()) %>% 
  group_by(source, year, month) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, month, "1", sep="-"))) %>% 
  ungroup() %>%
  # select(macro_type) %>% distinct()
  filter(year > 2014, year != 2021,
         str_detect(macro_type,"market")) %>% 
  # select(source) %>% distinct()
  ggplot(aes(x=date, y=per, group=source, color=source)) +
  geom_line( size=1, alpha=0.7) +
  ggtitle("Use of Market Indicators (By Month)") +
  xlab("Date") +
  ylab("Proportion")

quant_data %>% 
  left_join(article_data, by=c("article_id"="id")) %>% 
  filter(str_detect(frame, "macro"),
         str_detect(type, "macro"),
         str_detect(source, "wsj|nyt|wash"),
         str_detect(macro_type,"price")) %>% 
  mutate(year = year(date),
         month = month(date)) %>% 
  group_by(source, year, month, macro_type, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, year, month, macro_type) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, month, "1", sep="-"))) %>% 
  ungroup() %>%
  # select(macro_type) %>% distinct()
  filter(year > 2014, year != 2021, str_detect(spin, "neg"), count > 1) %>% 
  # select(source) %>% distinct()
  ggplot(aes(x=date, y=per, group=source, color=source)) +
  geom_line( size=1, alpha=0.7) +
  ggtitle("Negativity of Price Indicators (By Month)") +
  xlab("Date") +
  ylab("Proportion")

quant_data %>% 
  left_join(article_data, by=c("article_id"="id")) %>% 
  filter(str_detect(frame, "macro"),
         str_detect(type, "macro"),
         !str_detect(source, "bbc")
         ) %>% 
  mutate(year = year(date),
         month = month(date)) %>%
  filter(year < 2023) %>% 
  group_by(source, year, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, year) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, "1-1", sep="-"))) %>% 
  ungroup() %>%
  # select(macro_type) %>% distinct()
  filter(year > 2014, year != 2021, str_detect(spin, "neg"), count > 1) %>%
  # select(source) %>% distinct()
  ggplot(aes(x=date, y=per, group=source, color=source)) +
  geom_line( size=1, alpha=0.7) +
  ggtitle("Negativity of Indicators (By Year)") +
  xlab("Date") +
  ylab("Proportion")

quant_data %>% 
  left_join(article_data, by=c("article_id"="id")) %>% 
  filter(str_detect(frame, "macro"),
         str_detect(type, "macro"),
         !str_detect(source, "bbc")
  ) %>% 
  head()
  mutate(year = year(date),
         month = month(date)) %>%
  filter(year < 2023) %>% 
  group_by(source, year, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, year) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, "1-1", sep="-"))) %>% 
  ungroup() %>%
  # select(macro_type) %>% distinct()
  filter(year > 2014, year != 2021, str_detect(spin, "neg"), count > 1) %>%
  # select(source) %>% distinct()
  ggplot(aes(x=date, y=per, group=source, color=source)) +
  geom_line( size=1, alpha=0.7) +
  ggtitle("Negativity of Indicators (By Year)") +
  xlab("Date") +
  ylab("Proportion")

#
# Presidential
#
# Define presidential terms
presidential_periods <- tibble(
  president = c("Obama", "Trump", "Biden"),
  start_date = as.Date(c("2015-01-01", "2017-01-20", "2021-01-20")),
  end_date = as.Date(c("2017-01-19", "2021-01-19", "2022-12-31"))
)

# Modified analysis code
quant_data %>% 
  left_join(article_data, by=c("article_id"="id")) %>% 
  filter(str_detect(frame, "macro"),
         str_detect(type, "macro"),
         !str_detect(source, "bbc")
  ) %>% 
  # Add presidential period
  mutate(
    president = case_when(
      date >= as.Date("2015-01-01") & date < as.Date("2017-01-20") ~ "Obama",
      date >= as.Date("2017-01-20") & date < as.Date("2021-01-20") ~ "Trump",
      date >= as.Date("2021-01-20") ~ "Biden",
      TRUE ~ "Other"
    )
  ) %>%
  # Calculate negativity rates
  group_by(source, president, spin) %>%
  summarise(count = n(), .groups = "keep") %>%
  group_by(source, president) %>%
  mutate(
    total = sum(count),
    proportion = count / total
  ) %>%
  ungroup() %>%
  # Filter for negative sentiment and valid counts
  filter(spin == "neg", count > 1) %>%
  # Create visualization
  ggplot(aes(x = president, y = proportion, color = source)) +
  geom_point(size = 3, alpha = 0.6) +
  geom_line(aes(group = source), size = 1, alpha = 0.6) +
  # Add mean line across sources
  stat_summary(
    aes(group = president),
    fun = mean,
    geom = "point",
    size = 5,
    color = "black",
    shape = 18
  ) +
  # Customize theme and labels
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 0, size = 10),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  ) +
  labs(
    title = "Negativity Rate in Economic Coverage by Presidential Administration",
    x = "President",
    y = "Proportion of Negative Sentiment",
    color = "News Source"
  ) +
  scale_y_continuous(
    labels = scales::percent_format(),
    limits = c(0, NA)
  )
  
quant_data %>% 
  left_join(article_data, by=c("article_id"="id")) %>% 
  filter(str_detect(frame, "macro"),
         str_detect(type, "macro"),
         !str_detect(source, "bbc")
  ) %>% 
  # Add presidential period with COVID period
  mutate(
    president = case_when(
      # COVID period takes precedence during its timeframe
      date >= as.Date("2020-03-15") & date < as.Date("2021-01-20") ~ "COVID",
      date >= as.Date("2015-01-01") & date < as.Date("2017-01-20") ~ "Obama",
      date >= as.Date("2017-01-20") & date < as.Date("2020-03-15") ~ "Trump",  # Trump pre-COVID
      date >= as.Date("2021-01-20") ~ "Biden",
      TRUE ~ "Other"
    ),
    # Create ordered factor with chronological order including COVID
    president = factor(president, 
                       levels = c("Obama", "Trump", "COVID", "Biden"),
                       ordered = TRUE)
  ) %>%
  group_by(source, president, spin) %>%
  summarise(count = n(), .groups = "keep") %>%
  group_by(source, president) %>%
  mutate(
    total = sum(count),
    proportion = count / total
  ) %>%
  ungroup() %>%
  filter(spin == "neg", count > 1,
         !is.na(president)) %>%
  ggplot(aes(x = president, y = proportion, color = source)) +
  geom_point(size = 3, alpha = 0.6) +
  geom_line(aes(group = source), size = 1, alpha = 0.4) +
  stat_summary(
    aes(group = president),
    fun = mean,
    geom = "point",
    size = 5,
    color = "black",
    shape = 18
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 0, size = 10),
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5)
  ) +
  labs(
    title = "Negativity Rate in Economic Coverage by Period",
    # subtitle = "COVID period: March 15, 2020 - January 20, 2021",
    x = "Period",
    y = "Proportion of Negative Sentiment",
    color = "News Source"
  ) +
  scale_y_continuous(
    labels = scales::percent_format(),
    limits = c(0, NA)
  )


# Prop inflation articles

get_price_per = function(data, source_name) {
  data %>% 
    left_join(article_data, by=c("article_id"="id")) %>% 
    filter(str_detect(frame, "macro"),
           str_detect(type, "macro"),
           str_detect(source, source_name)) %>% 
    mutate(year = year(date),
           month = month(date)) %>%
    group_by(article_id, macro_type, year, month) %>%
    summarise(count = n()) %>% 
    mutate(price = if_else(str_detect(macro_type, "price"), 1, 0)) %>% 
    group_by(article_id, year, month) %>% 
    summarise(is_price = sum(price)) %>% 
    group_by(year, month) %>% 
    summarise(tot = n(),
              price_count = sum(is_price)) %>% 
    mutate(per_price = price_count / tot,
           date = as_date(paste(year, month, "1", sep="-")),
           source = source_name) %>% 
    ungroup() %>% 
    select(date, source, per_price)
}

wp_p = get_price_per(quant_data, "wash")
wsj_p = get_price_per(quant_data, "wsj")
nyt_p = get_price_per(quant_data, "nyt")

wp_p %>% 
  rbind(wsj_p) %>%
  rbind(nyt_p) %>% 
  pivot_wider(names_from = source, values_from = per_price) %>% 
  filter(date > "2014-12-31") %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y=nyt, color="nyt")) + #, color="blue") +
  geom_line(aes(y=wash, color="wapo")) + #, color="orange") +
  geom_line(aes(y=wsj, color="wsj")) + #, color="black") +
  ggtitle("% of Articles With Inflation Datapoints") +
  xlab("Date") +
  ylab("Proportion")

quant_data %>% 
  left_join(article_data, by=c("article_id"="id")) %>% 
  filter(str_detect(frame, "macro"),
         str_detect(type, "macro"),
         str_detect(source, "wsj")) %>% 
  mutate(year = year(date),
         month = month(date)) %>%
  group_by(article_id, macro_type, year, month) %>%
  summarise(count = n()) %>% 
  mutate(price = if_else(str_detect(macro_type, "price"), 1, 0)) %>% 
  group_by(article_id, year, month) %>% 
  summarise(is_price = sum(price)) %>% 
  group_by(year, month) %>% 
  summarise(tot = n(),
            price_count = sum(is_price)) %>% 
  mutate(per_price = price_count / tot,
         date = as_date(paste(year, month, "1", sep="-"))) %>% 
  ungroup() %>% 
  ggplot(aes(x=date, y=per_price)) +
  geom_line()

