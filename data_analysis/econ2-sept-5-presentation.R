article_level_annotations %>% head()

#
# Volume of reporting
#

article_level_annotations %>%
  mutate(frame_prediction = as.factor(frame_prediction),
         year = year(date)) %>%
  filter(year > 2015, year < 2023) %>% 
  group_by(publisher, year, frame_prediction) %>% 
  summarise(count = n()) %>% 
  group_by(publisher, year) %>% 
  mutate(total = sum(count)) %>% 
  ggplot(aes(x=year, y=total, group=publisher, fill=publisher)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Volume of Economic Reporting Over Time")

article_level_annotations %>%
  mutate(frame_prediction = as.factor(frame_prediction),
         year = year(date)) %>%
  filter(year > 2015, year < 2023, str_detect(frame_prediction, "macro")) %>% 
  group_by(publisher, year, frame_prediction) %>% 
  summarise(count = n()) %>% 
  group_by(publisher, year) %>% 
  mutate(total = sum(count)) %>% 
  ggplot(aes(x=year, y=total, group=publisher, fill=publisher)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Volume of Macro-Economic Reporting Over Time")

#
# Article level frames
#

article_level_annotations %>%
  mutate(frame_prediction = as.factor(frame_prediction),
         year = year(date)) %>%
  filter(year > 2015, year < 2023) %>% 
  # group_by(publisher, year, frame_prediction) %>% 
  # summarise(count = n()) %>% 
  # group_by(publisher, year) %>% 
  # mutate(total = sum(count)) %>% 
  ggplot(aes(x=year, group=publisher, fill=publisher)) +
  geom_bar(position = "dodge") +
  facet_wrap(~frame_prediction) +
  scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("High Level Frames Can Shift Over Time")

article_level_annotations %>%
  mutate(frame_prediction = as.factor(frame_prediction),
         year = year(date)) %>%
  filter(year > 2015, year < 2023) %>% 
  group_by(publisher, frame_prediction) %>% 
  summarise(count = n()) %>% 
  group_by(publisher) %>% 
  mutate(total = sum(count),
         per= count/total) %>% 
  ggplot(aes(x=frame_prediction, y=per, group=publisher, fill=publisher)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values=cbbPalette) +
  theme_bw() + 
  ggtitle("Publishers Favor Different High Level Frames")

article_level_annotations %>%
  mutate(frame_prediction = as.factor(frame_prediction),
         year = year(date)) %>%
  filter(year > 2015, year < 2023) %>% 
  group_by(publisher, year, frame_prediction) %>% 
  summarise(count = n()) %>% 
  group_by(publisher, year) %>% 
  mutate(total = sum(count),
         per= count/total) %>% 
  ggplot(aes(x=year, y=per, group=publisher, fill=publisher)) +
  geom_col(position = "dodge") +
  facet_wrap(~frame_prediction) +
  scale_fill_manual(values=cbbPalette) +
  theme_bw() + 
  ggtitle("Publishers Lean Into Different High Level Frames Over Time")

#
# Quant use
#

joined_data %>% 
  select(macro_type) %>% distinct()

joined_data %>% 
  filter(!is.na(frame_prediction),
         !is.na(macro_type),
         str_detect(macro_type, "jobs|retail|energy|wages|macro|housing|prices")) %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         year = year(date.x)) %>%
  filter(year > 2015, year < 2023) %>% 
  ggplot(aes(x=macro_type, group=source, fill=source)) +
  geom_bar(position = "dodge") +
  facet_wrap(~year) +
  scale_fill_manual(values=cbbPalette) +
  theme_bw()


joined_data %>% 
  filter(!is.na(frame_prediction),
         !is.na(macro_type),
         str_detect(macro_type, "jobs|retail|energy|wages|macro|housing|prices|interest"),
         str_detect(frame_prediction, "macro")) %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         year = year(date.x)) %>%
  filter(year > 2015, year < 2023) %>% 
  group_by(year, macro_type) %>% 
  summarise(count = n()) %>% 
  group_by(year) %>% 
  mutate(total = sum(count),
         per= count/total) %>% 
  ggplot(aes(x=year, y=per, group=macro_type, fill=macro_type)) +
  geom_col(position = "dodge") +
  # facet_wrap(~year) +
  scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Indicator Use Shifts Over Time")

joined_data %>% 
  filter(!is.na(frame_prediction),
         !is.na(macro_type),
         str_detect(macro_type, "jobs|retail|energy|wages|macro|housing|prices|interest"),
         str_detect(frame_prediction, "macro")) %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         year = year(date.x)) %>%
  filter(year > 2015, year < 2023) %>% 
  group_by(source, year, macro_type) %>% 
  summarise(count = n()) %>% 
  group_by(source, year) %>% 
  mutate(total = sum(count),
         per= count/total) %>% 
  ggplot(aes(x=year, y=per, group=macro_type, fill=macro_type)) +
  geom_col(position = "dodge") +
  facet_wrap(~source) +
  scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("These Changes Vary Across Publishers")

#
# Article Level Neg Bias
#

article_level_annotations %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         econ_rate_prediction = as.factor(econ_rate_prediction),
         year = year(date)) %>%
  filter(year > 2015, year < 2023,
         str_detect(econ_rate_prediction, "good|poor"),
         str_detect(frame_prediction, "macro")) %>% 
  group_by(publisher, year, econ_rate_prediction) %>% 
  summarise(count = n()) %>% 
  group_by(publisher, year) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  ggplot(aes(x=econ_rate_prediction, y=per, group=publisher, fill=publisher)) +
  geom_col(position = "dodge") +
  facet_wrap(~year) +
  scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Article Level Negativity Bias (Macro Articles Only)")

article_level_annotations %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         econ_rate_prediction = as.factor(econ_rate_prediction),
         year = year(date)) %>%
  filter(year > 2015, year < 2023,
         str_detect(econ_rate_prediction, "good|poor"),
         str_detect(frame_prediction, "macro")) %>% 
  group_by(publisher, year, econ_rate_prediction) %>% 
  summarise(count = n()) %>% 
  group_by(publisher, year) %>% 
  mutate(total = sum(count),
         per = count / total) %>%
  filter(str_detect(econ_rate_prediction, "poor")) %>% 
  ggplot(aes(x=year, y=per)) +
  geom_line(aes(group=publisher, color=publisher), size=1) +
  # geom_smooth(alpha = 0.5) + 
  scale_color_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Article Level Negativity Bias (Macro Articles Only)")

#
# Quant Neg Bias
#

joined_data %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         econ_rate_prediction = as.factor(econ_rate_prediction),
         year = year(date.x)) %>%
  filter(year > 2015, year < 2023,
         str_detect(frame_prediction, "macro"),
         !is.na(frame_prediction),
         !is.na(macro_type),) %>% 
  group_by(source, year, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, year) %>% 
  mutate(total = sum(count),
         per= count/total) %>% 
  filter(str_detect(spin, "neg|pos")) %>% 
  ggplot(aes(x=spin, y=per)) +
  geom_col(aes(group=source, fill=source), position = "dodge") +
  facet_wrap(~year) + 
  scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Quantity Level Negativity Bias (Macro Articles Only)")


joined_data %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         econ_rate_prediction = as.factor(econ_rate_prediction),
         year = year(date.x)) %>%
  filter(year > 2015, year < 2023,
         str_detect(frame_prediction, "macro"),
         !is.na(frame_prediction),
         !is.na(macro_type),) %>% 
  group_by(source, year, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, year) %>% 
  mutate(total = sum(count),
         per= count/total) %>% 
  filter(str_detect(spin, "neg")) %>% 
  ggplot(aes(x=year, y=per)) +
  # geom_col(aes(group=source, fill=source), position = "dodge") +
  # facet_wrap(~year) + 
  geom_line(aes(group=source, color=source), size=1) +
  # geom_smooth(alpha = 0.5) + 
  scale_color_manual(values=cbbPalette) +
  # scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Quantity Level Negativity Bias (Macro Articles Only)")

joined_data %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         econ_rate_prediction = as.factor(econ_rate_prediction),
         year = year(date.x)) %>%
  filter(year > 2015, year < 2023,
         str_detect(frame_prediction, "macro"),
         !is.na(frame_prediction),
         !is.na(macro_type),) %>% 
  group_by(source, year, article_id, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, year, article_id) %>% 
  mutate(total = sum(count),
         per= count/total,
         is_neg = if_else(per > 0.5, 1, 0)) %>% 
  filter(str_detect(spin, "neg")) %>%
  group_by(source, year) %>% 
  summarise(count = n(),
            neg_count = sum(is_neg)) %>% 
  mutate(per_neg = neg_count / count) %>% 
  ggplot(aes(x=year, y=per_neg)) +
  geom_line(aes(group=source, color=source), size=1) +
  scale_color_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Percent of Articles With > 50% Negative Quantities (Macro Articles Only)")

#
# Article level negative by pres
#

article_level_annotations %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         econ_rate_prediction = as.factor(econ_rate_prediction),
         year = year(date),
         president = if_else(year < 2017, "Obama", 
                             if_else(year >= 2017 & year < 2020, "Trump",
                                     if_else(year == 2020, "COVID", 
                                     if_else(year >= 2021, "Biden", "NA")))),
         president = factor(president, levels = c("Obama", "Trump", "COVID", "Biden"))) %>%
  filter(str_detect(econ_rate_prediction, "good|poor"),
         str_detect(frame_prediction, "macro")) %>% 
  group_by(publisher, president, econ_rate_prediction) %>% 
  summarise(count = n()) %>% 
  group_by(publisher, president) %>% 
  mutate(total = sum(count),
         per = count/total) %>% 
  ggplot(aes(x=econ_rate_prediction, y=per, group=publisher, fill=publisher)) +
  geom_col(position = "dodge") +
  facet_wrap(~president) +
  scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Article Level Negativity Bias By Presidential Administration (Macro Articles Only)")
  
article_level_annotations %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         econ_rate_prediction = as.factor(econ_rate_prediction),
         year = year(date),
         president = if_else(year < 2017, "Obama", 
                             if_else(year >= 2017 & year < 2020, "Trump",
                                     if_else(year == 2020, "COVID", 
                                             if_else(year >= 2021, "Biden", "NA")))),
         president = factor(president, levels = c("Obama", "Trump", "COVID", "Biden"))) %>%
  filter(str_detect(econ_rate_prediction, "good|poor"),
         str_detect(frame_prediction, "macro")) %>% 
  group_by(publisher, president, econ_rate_prediction) %>% 
  summarise(count = n()) %>% 
  group_by(publisher, president) %>% 
  mutate(total = sum(count),
         per = count/total) %>% 
  filter(str_detect(econ_rate_prediction, "poor")) %>% 
  ggplot(aes(x=president, y=per, group=publisher, fill=publisher)) +
  geom_col(position = "dodge") +
  facet_wrap(~publisher) +
  scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Article Level Negativity Bias By Presidential Administration (Macro Articles Only)")

#
# Quantity bias by president
#

joined_data %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         econ_rate_prediction = as.factor(econ_rate_prediction),
         year = year(date.x),
         president = if_else(year < 2017, "Obama", 
                             if_else(year >= 2017 & year < 2020, "Trump",
                                     if_else(year == 2020, "COVID", 
                                             if_else(year >= 2021, "Biden", "NA")))),
         president = factor(president, levels = c("Obama", "Trump", "COVID", "Biden"))) %>%
  filter(year > 2015, year < 2023,
         str_detect(frame_prediction, "macro"),
         !is.na(frame_prediction),
         !is.na(macro_type),) %>% 
  group_by(source, president, article_id, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, president, article_id) %>% 
  mutate(total = sum(count),
         per= count/total,
         is_neg = if_else(per > 0.5, 1, 0)) %>% 
  filter(str_detect(spin, "neg")) %>%
  group_by(source, president) %>% 
  summarise(count = n(),
            neg_count = sum(is_neg)) %>% 
  mutate(per_neg = neg_count / count) %>% 
  ggplot(aes(x=president, y=per_neg)) +
  geom_col(aes(group=source, fill=source), position = "dodge") +
  scale_fill_manual(values=cbbPalette) +
  facet_wrap(~source) +
  theme_bw() +
  ggtitle("Percent of Articles With > 50% Negative Quantities (Macro Articles Only)")


joined_data %>% 
  mutate(frame_prediction = as.factor(frame_prediction),
         econ_rate_prediction = as.factor(econ_rate_prediction),
         year = year(date.x),
         president = if_else(year < 2017, "Obama", 
                             if_else(year >= 2017 & year < 2020, "Trump",
                                     if_else(year == 2020, "COVID", 
                                             if_else(year >= 2021, "Biden", "NA")))),
         president = factor(president, levels = c("Obama", "Trump", "COVID", "Biden"))) %>%
  filter(year > 2015, year < 2023,
         str_detect(frame_prediction, "macro"),
         !is.na(frame_prediction),
         !is.na(macro_type),) %>% 
  group_by(source, president, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, president) %>% 
  mutate(total = sum(count),
         per= count/total) %>% 
  filter(str_detect(spin, "neg|pos")) %>% 
  ggplot(aes(x=spin, y=per)) +
  geom_col(aes(group=source, fill=source), position = "dodge") +
  facet_wrap(~president) + 
  scale_fill_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Quantity Level Negativity Bias (Macro Articles Only)")

#
# Underlying Data
#

# CPI

cpi_avg = cpi %>% 
  mutate(date = floor_date(as_datetime(DATE), unit="quarter")) %>% 
  group_by(date) %>% 
  summarise(cpi = mean(USACPALTT01CTGYM))


joined_data %>% 
  filter(str_detect(frame_prediction, "macro"),
         str_detect(macro_type, "price")) %>% 
  mutate(date = floor_date(date.x, unit="quarter"),
         year = year(date)) %>% 
  filter(year > 2015, year < 2023) %>%
  group_by(source, date, macro_type, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, date, macro_type) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  left_join(cpi_avg, by = c("date"="date")) %>%
  filter(str_detect(spin, "neg")) %>% 
  group_by(source) %>% 
  mutate(max_neg = max(count),
         relative = count / max_neg) %>% 
  ggplot(aes(x=date, y=relative, group=source, color=source)) +
  geom_line(aes(y=cpi / 8), linetype=2, color = "gray", size=0.7) +
  geom_line(size=1) +
  # ylim(0,100) +
  facet_wrap(~source) +
  scale_color_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("CPI vs Price Quantity Negativity")  

# Jobs

bls_avg =  bls_nonfarm %>% 
  filter(!is.na(diff)) %>% 
  mutate(date = floor_date(as_datetime(ym), unit="quarter")) %>% 
  group_by(date) %>% 
  summarise(mean_diff = mean(diff))

joined_data %>% 
  filter(str_detect(frame_prediction, "macro"),
         str_detect(macro_type, "job")) %>% 
  mutate(date = floor_date(date.x, unit="quarter"),
         year = year(date)) %>% 
  filter(year > 2015) %>%
  group_by(source, date, macro_type, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, date, macro_type) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  left_join(bls_avg, by = c("date"="date")) %>%
  filter(str_detect(spin, "neg")) %>% 
  group_by(source) %>% 
  mutate(max_neg = max(count),
         relative = count / max_neg) %>% 
  ggplot(aes(x=date, y=relative, group=source, color=source)) +
  geom_line(aes(y=mean_diff / 1000), linetype=2, color = "gray", size=0.7) +
  geom_line(size=1) +
  # ylim(-2, 2) +
  coord_cartesian(xlim = c(as_datetime("2015-01-01"), as_datetime("2022-10-01")), ylim = c(0,1)) +
  facet_wrap(~source) +
  scale_color_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Change in Non-Farm Payroll vs Job Quantity Negativity") 

joined_data %>% 
  filter(str_detect(frame_prediction, "macro"),
         str_detect(macro_type, "job")) %>% 
  mutate(date = floor_date(date.x, unit="quarter"),
         year = year(date)) %>% 
  filter(year > 2015, year < 2023) %>%
  group_by(source, date, macro_type, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, date, macro_type) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  left_join(bls_avg, by = c("date"="date")) %>%
  filter(str_detect(spin, "neg")) %>% 
  group_by(source) %>% 
  mutate(max_neg = max(count),
         relative = count / max_neg) %>% 
  ggplot(aes(x=date, y=relative, group=source, color=source)) +
  geom_line(aes(y=mean_diff / 1000), linetype=2, color = "gray", size=0.7) +
  geom_line(size=1) +
  # ylim(-2, 2) +
  coord_cartesian(xlim = c(as_datetime("2015-01-01"), as_datetime("2022-10-01")), ylim = c(0,1)) +
  facet_wrap(~source) +
  scale_color_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Change in Non-Farm Payroll vs Job Quantity Negativity")  

#
# Negativity vs data
#


joined_data %>% 
  filter(str_detect(frame_prediction, "macro"),
         str_detect(macro_type, "job")) %>% 
  mutate(date = floor_date(date.x, unit="quarter"),
         year = year(date)) %>% 
  filter(year > 2015, year < 2023) %>%
  group_by(source, date, macro_type, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, date, macro_type) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  left_join(bls_avg, by = c("date"="date")) %>%
  # head()
  # group_by(source) %>%
  
  # mutate(max_neg = max(count),
  #        relative = count / max_neg) %>% 
  filter(str_detect(spin, "neg")) %>% 
  ggplot(aes(x=date, y=per, group=source, color=source)) +
  geom_line(aes(y=mean_diff / 1000), linetype=2, color = "gray", size=0.7) +
  geom_line(size=1) +
  # ylim(-2, 2) +
  coord_cartesian(xlim = c(as_datetime("2015-01-01"), as_datetime("2022-10-01")), ylim = c(0,1)) +
  facet_wrap(~source) +
  scale_color_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("Change in Non-Farm Payroll vs Job Quantity Negativity (%)")  

# cpi



joined_data %>% 
  filter(str_detect(frame_prediction, "macro"),
         str_detect(macro_type, "price")) %>% 
  mutate(date = floor_date(date.x, unit="quarter"),
         year = year(date)) %>% 
  filter(year > 2015, year < 2023) %>%
  group_by(source, date, spin) %>% 
  summarise(count = n()) %>% 
  group_by(source, date) %>% 
  mutate(total = sum(count),
         per = count / total) %>% 
  left_join(cpi_avg, by = c("date"="date")) %>%
  filter(str_detect(spin, "neg")) %>% 
  # group_by(source) %>% 
  # mutate(max_neg = max(count),
  #        relative = count / max_neg) %>% 
  ggplot(aes(x=date, y=per, group=source, color=source)) +
  geom_line(aes(y=cpi / 8), linetype=2, color = "gray", size=0.7) +
  geom_line(size=1) +
  # ylim(0,100) +
  facet_wrap(~source) +
  scale_color_manual(values=cbbPalette) +
  theme_bw() +
  ggtitle("CPI vs Price Quantity Negativity (%)")  

