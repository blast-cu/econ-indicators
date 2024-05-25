#
# STARTING POINT
#
article_spin_jobs %>%
  filter(str_detect(source, "wash"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  # REPORTING PER
  geom_col(data=jobs_reporting_per %>% filter(str_detect(source, "wash"), str_detect(macro_type, "jobs")),
            aes(x=yq, y=per, colour="% of Reporting"), fill="white", width=25) + #, linetype = "dotdash", size=1
  # JOBS
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("Change in Payroll" ="#000000", "% of Reporting"="darkgray", #    #CC79A7
                                "washingtonpost.negative"="#E69F00", "washingtonpost.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("% Negative", "% Positive",
                                "% of Reporting", "Change in Payroll")) +
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1))  +
  ggtitle("Article Level Spin for Jobs Articles in the Washington Post (Quarterly)") +
  xlab("Date") +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll (Thousands)")
  ) + 
  theme_bw() +
  theme(legend.position = 'bottom')

article_spin_jobs %>%
  filter(str_detect(source, "nytimes"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "nyt"), str_detect(macro_type, "jobs")),
           aes(x=yq, y=per, colour="% of Reporting"), linetype = "dotdash", size=1) + #
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("Change in Payroll" ="#000000", "% of Reporting"="#CC79A7", #    
                                "nytimes.negative"="#E69F00", "nytimes.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("% Negative", "% Positive",
                                "% of Reporting", "Change in Payroll")) +
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("Article Level Spin for Jobs Articles in the New York Times (Quarterly)") +
  xlab("Date") +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll (Thousands)")
  ) + 
  theme_bw()


article_spin_jobs %>%
  filter(str_detect(source, "nytimes"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash")   +
  geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / 8.97, colour="CPI (% YOY)")) +
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "nyt"), str_detect(macro_type, "jobs")), aes(x=yq, y=per, colour="% of Reporting (Jobs)"), linetype = "dotdash", size=1) +
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "nyt"), str_detect(macro_type, "prices+")), aes(x=yq, y=per, colour="% of Reporting (Prices+)"), linetype = "dotdash", size=1) +
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("% Jobs & Prices+ Reporting in the New York Times (Quarterly)") +
  xlab("Date") +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll")
  ) + 
  theme_bw()

article_spin_jobs %>%
  filter(str_detect(source, "nytimes"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  # WSJ
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "wsj"), str_detect(macro_type, "job")), 
            aes(x=yq, y=per, colour="% of Reporting (WSJ, Jobs)"), size=0.5) +
  # WAPO
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "wash"), str_detect(macro_type, "job")), 
            aes(x=yq, y=per, colour="% of Reporting (WAPO, Jobs)"), size=0.5) +
  # JOBS
  geom_line(data=bls_nonfarm, aes(x = ym, y = 0.5+(diff) / 5000, colour="Change in Payroll"), linetype="twodash") + 
  # NYT
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "nyt"), str_detect(macro_type, "job")), 
            aes(x=yq, y=per, colour="% of Reporting (NYT, Jobs)"), size=1) +
  scale_color_manual(values = c("% of Reporting (WSJ, Jobs)" ="lightgray", "% of Reporting (WAPO, Jobs)"="darkgray", 
                                "% of Reporting (NYT, Jobs)"="#E69F00", "Change in Payroll"="#56B4E9"),
                     name="Colour",
                     labels = c("% of Reporting (NYT, Jobs)",  "% of Reporting (WSJ, Jobs)", 
                                "% of Reporting (WAPO, Jobs)", "CPI (% YOY)")) +
  coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("% Jobs Reporting in the New York Times vs Payroll Changes (Quarterly)") +
  xlab("Date") +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.5) * 5000 , name = "Change in Non-Farm Payroll (Thousands")
  ) + 
  theme_bw()



article_spin_jobs %>%
  filter(str_detect(source, "nytimes"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  # WSJ
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "wsj"), str_detect(macro_type, "prices+")), 
            aes(x=yq, y=per, colour="% of Reporting (WSJ, Prices+)"), size=0.5) +
  # WAPO
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "wash"), str_detect(macro_type, "prices+")), 
            aes(x=yq, y=per, colour="% of Reporting (WAPO, Prices+)"), size=0.5) +
  geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / (4*8.97), colour="CPI (% YOY)"), linetype = "dotdash", size=1) +
  # NYT
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "nyt"), str_detect(macro_type, "prices+")), 
            aes(x=yq, y=per, colour="% of Reporting (NYT, Prices+)"), size=1) +
  scale_color_manual(values = c("% of Reporting (WSJ, Prices+)" ="lightgray", "% of Reporting (WAPO, Prices+)"="darkgray", 
                                "% of Reporting (NYT, Prices+)"="#E69F00", "CPI (% YOY)"="#56B4E9"),
                     name="Colour",
                     labels = c("% of Reporting (NYT, Prices+)",  "% of Reporting (WAPO, Prices+)", 
                                "% of Reporting (WSJ, Prices+)", "CPI (% YOY)")) +
  # coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("% Prices+ Reporting in the New York Times vs CPI Rate (Quarterly)") +
  xlab("Date") +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.0) * (4*8.97) , name = "CPI (% Change YOY)")
  ) + 
  theme_bw()





jobs_reporting_per %>% 
  filter(str_detect(macro_type, "prices+"), str_detect(source, "nyt|wash|wsj")) %>% 
  ggplot() +
  geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / (4*8.97), colour="CPI (% YOY)"), linetype = "dotdash", size=1) +
  geom_line(aes(x=yq, y=per, colour=source, group=source), size=1) +
  ggtitle("% Prices+ Reporting by Source vs CPI Rate (Quarterly)") +
  xlab("Date") +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.0) * (4*8.97) , name = "CPI (% Change YOY)")
  ) + 
  theme_bw()
#"% of Reporting (Prices+)"

article_spin_prices %>%
  filter(str_detect(source, "nyt"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / (4*8.97), colour="CPI (% YOY)")) +
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "nyt"), str_detect(macro_type, "prices+")), 
            aes(x=yq, y=per, colour="% of Reporting (Prices+)"), linetype = "dotdash", size=1) +
  # coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  ggtitle("Article Level Spin for Prices+ Articles in the New York Times (Quarterly)") +
  xlab("Date") +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.0) * (4*8.97) , name = "CPI (% Change YOY)")
  ) + 
  theme_bw()


article_spin_prices %>%
  filter(str_detect(source, "wash"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / (4*8.97), colour="CPI (% YOY)")) +
  geom_line(data=jobs_reporting_per %>% filter(str_detect(source, "wash"), str_detect(macro_type, "prices+")), 
            aes(x=yq, y=per, colour="% of Reporting (Prices+)"), linetype = "dotdash", size=1) +
  # coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  ggtitle("Article Level Spin for Prices+ Articles in the WAPO (Quarterly)") +
  xlab("Date") +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.0) * (4*8.97) , name = "CPI (% Change YOY)")
  ) + 
  theme_bw()

# Look at overall pos vs neg trends 


article_level_spin = macro_data %>% 
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
  filter(total > 1) %>%
  group_by(source, yq, article_spin) %>% 
  summarise(count = n()) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  ungroup()

article_level_spin %>% 
  filter(str_detect(source, "wsj")) %>% 
  ggplot() +
  geom_line(aes(x=yq, y=per, group=article_spin, color=article_spin))

jobs_reporting_per_plus_spin %>% 
  filter(str_detect(source, "nyt"), str_detect(macro_type, "price")) %>% 
  ggplot() +
  geom_line(aes(x=yq, y=per, color=spin))


per_price_article = macro_data %>% 
  mutate(macro_type = if_else(str_detect(macro_type, "energy|prices"), "prices+", macro_type)) %>% 
  group_by(source, date, article_id) %>% 
  summarise(has_neg_price = sum(str_detect(macro_type, "prices") & spin == "negative"),
            has_pos_price = sum(str_detect(macro_type, "prices") & spin == "positive"),
            has_price = sum(str_detect(macro_type, "prices"))) %>% 
  ungroup() %>% 
  mutate(year = year(date),
         month = month(date),
         quarter = as.integer(floor(month / 3)),
         ym = ym(paste(year, month, sep="-")),
         yq = yq(paste(year, quarter, sep="."))) %>% 
  group_by(source, yq) %>% 
  summarise(count = n(),
            count_price = sum(has_price > 0),
            count_neg_price = sum(has_neg_price > 0),
            count_pos_price = sum(has_pos_price > 0),
            count_all_neg_price = sum(has_pos_price == 0 & has_price > 0)) %>% 
  mutate(per_price = count_price / count,
         per_neg_price = count_neg_price / count,
         per_all_neg_price = count_all_neg_price / count) 


per_price_article %>% 
  filter(str_detect(source, "nyt")) %>%
  ggplot() +
  geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / (2*8.97), colour="CPI (% YOY)"), linetype = "dotdash", size=1) +
  # NYT
  geom_line(aes(x=yq, y=per_price, color="% of Articles Mentioning Prices")) +
  geom_line(aes(x=yq, y=per_all_neg_price, color="% of Articles With Only Negative Prices")) +
  scale_color_manual(values = c("% of Articles With Only Negative Prices" ="#CC79A7", 
                                "% of Articles Mentioning Prices"="#E69F00", "CPI (% YOY)"="#56B4E9"),
                     name="Colour",
                     labels = c("% of Articles With Only Negative Prices",  "% of Articles Mentioning Prices", 
                                "CPI (% YOY)")) +
  # coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  ggtitle("% of Articles Mentioning Prices+ in the New York Times vs CPI Rate (Quarterly)") +
  xlab("Date") +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.0) * (2*8.97) , name = "CPI (% Change YOY)")
  ) + 
  theme_bw()


per_price_article %>% 
  filter(str_detect(source, "nyt")) %>% 
  ggplot(aes(x=yq)) +
  geom_line(aes(y=per_price, color="red")) +
  geom_line(aes(y=per_all_neg_price, color="blue"))

per_price_article %>%
  filter(str_detect(source, "nyt|wash|wsj")) %>% 
  mutate(pp_neg = per_all_neg_price / per_price) %>% 
  ggplot(aes(x=pp_neg, group=source,fill=source)) +
  geom_density(alpha=0.3) + 
  ggtitle("Density of % of Articles With 0 Positive Prices / % of Articles Mentioning Prices (Quarterly, by Source)")
