# LOAD DATA
setwd("~/Documents/econ-indicators/data-analysis")
macro_data = read_delim("data/macro_quant_annotations.csv")

macro_data = macro_data %>% 
  mutate(spin = if_else(spin == "pos", "positive", spin),
         spin = if_else(spin == "neg", "negative", spin))

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



# PREP DFS
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
  filter(str_detect(macro_type, "jobs"))

article_spin_jobs = macro_data %>% 
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
  group_by(source, yq, article_spin) %>% 
  summarise(count = n()) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  ungroup()

#GRAPH

# NYT
article_spin_jobs %>%
  filter(str_detect(source, "nytimes"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "nyt")), aes(x=yq, y=per, colour="% of Reporting"), linetype = "dotdash", size=1) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("Change in Payroll" ="#000000", "% of Reporting"="#CC79A7", 
                                "nytimes.negative"="#E69F00", "nytimes.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("Change in Payroll", "% of Reporting", 
                                "% Negative", "% Positive")) +
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("Article Level Spin for Jobs Articles in the New York Times (Quarterly)") +
  xlab("Date") +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll")
  ) + 
  theme_bw()

# WAPO
article_spin_jobs %>%
  filter(str_detect(source, "wash"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "wash")), aes(x=yq, y=per, colour="% of Reporting"), linetype = "dotdash", size=1) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("Change in Payroll" ="#000000", "% of Reporting"="#CC79A7", 
                                "washingtonpost.negative"="#E69F00", "washingtonpost.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("Change in Payroll", "% of Reporting", 
                                "% Negative", "% Positive")) +
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("Article Level Spin for Jobs Articles in the Washington Post (Quarterly)") +
  xlab("Date") +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll")
  ) + 
  theme_bw()

# WSJ
article_spin_jobs %>%
  filter(str_detect(source, "wsj"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "wsj")), aes(x=yq, y=per, colour="% of Reporting"), linetype = "dotdash", size=1) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("Change in Payroll" ="#000000", "% of Reporting"="#CC79A7", 
                                "wsj.negative"="#E69F00", "wsj.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("Change in Payroll", "% of Reporting", 
                                "% Negative", "% Positive")) +
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("Article Level Spin for Jobs Articles in the Wall Street Journal (Quarterly)") +
  xlab("Date") +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll")
  ) + 
  theme_bw()

# Fox
article_spin_jobs %>%
  filter(str_detect(source, "fox"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "fox")), aes(x=yq, y=per, colour="% of Reporting"), linetype = "dotdash", size=1) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("Change in Payroll" ="#000000", "% of Reporting"="#CC79A7", 
                                "foxnews.negative"="#E69F00", "foxnews.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("Change in Payroll", "% of Reporting", 
                                "% Negative", "% Positive")) +
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("Article Level Spin for Jobs Articles in Fox News (Quarterly)") +
  xlab("Date") +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll")
  ) + 
  theme_bw()


# HuffPo 
article_spin_jobs %>%
  filter(str_detect(source, "huff"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "huff")), aes(x=yq, y=per, colour="% of Reporting"), linetype = "dotdash", size=1) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("Change in Payroll" ="#000000", "% of Reporting"="#CC79A7", 
                                "huffpost.negative"="#E69F00", "huffpost.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("Change in Payroll", "% of Reporting", 
                                "% Negative", "% Positive")) +
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("Article Level Spin for Jobs Articles in HuffPost (Quarterly)") +
  xlab("Date") +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll")
  ) + 
  theme_bw()


# BBC 
article_spin_jobs %>%
  filter(str_detect(source, "bbc"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "bbc")), aes(x=yq, y=per, colour="% of Reporting"), linetype = "dotdash", size=1) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("Change in Payroll" ="#000000", "% of Reporting"="#CC79A7", 
                                "bbc.negative"="#E69F00", "bbc.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("Change in Payroll", "% of Reporting", 
                                "% Negative", "% Positive")) +
  coord_cartesian(xlim = c(as.Date("2018-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("Article Level Spin for Jobs Articles in the BBC (Quarterly)") +
  xlab("Date") +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll")
  ) + 
  theme_bw()


# Breitbart 
article_spin_jobs %>%
  filter(str_detect(source, "breit"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "breit")), aes(x=yq, y=per, colour="% of Reporting"), linetype = "dotdash", size=1) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("Change in Payroll" ="#000000", "% of Reporting"="#CC79A7", 
                                "breitbart.negative"="#E69F00", "breitbart.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("Change in Payroll", "% of Reporting", 
                                "% Negative", "% Positive")) +
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("Article Level Spin for Jobs Articles in Breitbart (Quarterly)") +
  xlab("Date") +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll")
  ) + 
  theme_bw()

macro_data %>%
  mutate(Year = year(date)) %>%
  filter(str_detect(macro_type, "jobs")) %>%
  group_by(Year, source, article_id) %>%
  summarise(count = n()) %>%
  group_by(Year, source) %>%
  summarise(count = n()) %>% view()

