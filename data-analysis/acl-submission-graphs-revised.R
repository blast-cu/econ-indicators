#
# JOB SPIN
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

# 
# CPI SPIN
#


article_spin_prices %>%
  filter(str_detect(source, "nyt"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  geom_col(data=jobs_reporting_per %>% filter(str_detect(source, "nyt"), str_detect(macro_type, "prices+")), 
           aes(x=yq, y=per, colour="% of Reporting (Prices+)"), fill="white", width = 25) +
  geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / (4*8.97), colour="CPI (% YOY)"), linetype="twodash") +
  # coord_cartesian(xlim = c(as.Date("2015-01-01"), as.Date("2022-10-01")), ylim = c(0,1)) +
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  ggtitle("Article Level Spin for Prices+ Articles in the New York Times (Quarterly)") +
  xlab("Date") +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("CPI (% YOY)" ="#000000", "% of Reporting (Prices+)"="darkgray", #    #CC79A7
                                "nytimes.negative"="#E69F00", "nytimes.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("% Negative", "% Positive",
                                "% of Reporting", "CPI (% YOY)")) +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.0) * (4*8.97) , name = "CPI (% Change YOY)")
  ) + 
  theme_bw() +
  theme(legend.position = 'bottom')



article_spin_prices %>%
  filter(str_detect(source, "wash"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  # PER REPORTING
  geom_col(data=jobs_reporting_per %>% filter(str_detect(source, "wash"), str_detect(macro_type, "prices+")), 
           aes(x=yq, y=per, colour="% of Reporting (Prices+)"), fill="white", width = 25) +
  # CPI 
  geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / (4*8.97), colour="CPI (% YOY)"), linetype="twodash") +
  # SPIN
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  ggtitle("Article Level Spin for Prices+ Articles in the Washington Post (Quarterly)") +
  xlab("Date") +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("CPI (% YOY)" ="#000000", "% of Reporting (Prices+)"="darkgray", #    #CC79A7
                                "washingtonpost.negative"="#E69F00", "washingtonpost.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("% Negative", "% Positive",
                                "% of Reporting", "CPI (% YOY)")) +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.0) * (4*8.97) , name = "CPI (% Change YOY)")
  ) + 
  theme_bw() +
  theme(legend.position = 'bottom')




article_spin_prices %>%
  filter(str_detect(source, "wsj"),
         !is.na(yq)) %>%
  filter(yq < as.Date("2023-01-01"),
         !str_detect(article_spin, "neutral")) %>% 
  ggplot() +
  # PER REPORTING
  geom_col(data=jobs_reporting_per %>% filter(str_detect(source, "wsj"), str_detect(macro_type, "prices+")), 
           aes(x=yq, y=per, colour="% of Reporting (Prices+)"), fill="white", width = 25) +
  # CPI 
  geom_line(data=cpi, aes(x=DATE, y=USACPALTT01CTGYM / (4*8.97), colour="CPI (% YOY)"), linetype="twodash") +
  # SPIN
  geom_line(size=1, aes(x=yq, y=per, group=interaction(source, article_spin), color=interaction(source, article_spin))) +
  ggtitle("Article Level Spin for Prices+ Articles in the Wall Street Journal (Quarterly)") +
  xlab("Date") +
  xlim(as.Date("2015-01-01"), as.Date("2022-10-01")) +
  scale_color_manual(values = c("CPI (% YOY)" ="#000000", "% of Reporting (Prices+)"="darkgray", #    #CC79A7
                                "wsj.negative"="#E69F00", "wsj.positive"="#56B4E9"),
                     name="Colour",
                     labels = c("% Negative", "% Positive",
                                "% of Reporting", "CPI (% YOY)")) +
  scale_y_continuous(
    "Percent", 
    sec.axis = sec_axis(~ (.-0.0) * (4*8.97) , name = "CPI (% Change YOY)")
  ) + 
  theme_bw() +
  theme(legend.position = 'bottom')



#
# CPI Selection
#


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
  theme_bw() +
  theme(legend.position = 'bottom')


#
# JOBS Selection
#

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
  theme_bw() +
  theme(legend.position = 'bottom')
