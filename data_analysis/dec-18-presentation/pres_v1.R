article_data %>%
  filter(str_detect(frame, "macro"),
         str_detect(source, "nyt|wsj")) %>%
  mutate(year = year(date),
         month = month(date)) %>%
  group_by(source, year, month) %>% 
  summarise(count = n()) %>% 
  group_by(source) %>% 
  mutate(max = max(count),
         relative_count = count / max,
         date = as_date(paste(year, month, "1", sep="-"))) %>% 
  filter(year > 2014 & !(str_detect(source, "wsj") & year==2021)) %>% 
  ggplot(aes(x=date, y=relative_count, group=source, color=source)) +
  geom_line() +
  theme_bw() +
  ggtitle("Macroeconomic Articles Per Month") +
  xlab("Date") +
  ylab("Relative Count")

article_data %>% 
  filter(str_detect(frame, "macro")) %>% 
  mutate(year = year(date),
         month = month(date),
         is_wsj = if_else(str_detect(source, "wsj"), "WSJ", "Non-WSJ")) %>% 
  group_by(is_wsj, year, month, econ_rate) %>% 
  summarise(count = n()) %>% 
  group_by(is_wsj, year, month) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, month, "1", sep="-"))) %>% 
  ungroup() %>%
  filter(str_detect(econ_rate, "poor"),
         year > 2014) %>% 
  ggplot(aes(x=date, y=per)) +
  geom_line(aes(color=is_wsj, group=is_wsj)) +
  # geom_smooth(se=FALSE) +
  ggtitle("% of Macroeconomic Articles That Were Negative (Per Month)")


article_data %>% 
  filter(str_detect(frame, "macro")) %>% 
  mutate(year = year(date),
         month = month(date),
         is_wsj = if_else(str_detect(source, "wsj"), "WSJ", "Non-WSJ")) %>% 
  group_by(is_wsj, year, econ_rate) %>% 
  summarise(count = n()) %>% 
  group_by(is_wsj, year) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, "1-1", sep="-"))) %>% 
  ungroup() %>%
  filter(str_detect(econ_rate, "poor"),
         year > 2014) %>% 
  ggplot(aes(x=date, y=per)) +
  geom_line(aes(color=is_wsj, group=is_wsj)) +
  # geom_smooth(se=FALSE) +
  ggtitle("% of Macroeconomic Articles That Were Negative (Annual)") +
  xlab("Date") +
  ylab("Proportion")

article_data %>% 
  filter(str_detect(frame, "macro")) %>% 
  mutate(year = year(date),
         month = month(date),
         is_wsj = if_else(str_detect(source, "wsj"), "WSJ", "Non-WSJ")) %>% 
  group_by(source, year, month, econ_rate) %>% 
  summarise(count = n()) %>% 
  group_by(source, year, month) %>% 
  mutate(tot = sum(count),
         per = count / tot) %>% 
  ungroup() %>%
  filter(str_detect(econ_rate, "poor"),
         year > 2014) %>% 
  group_by(year, month) %>% 
  summarise(max = max(per),
            min = min(per)) %>% 
  mutate(date = as_date(paste(year, month, "1", sep="-")),
         diff = max - min) %>% 
  ggplot(aes(x=date)) +
  geom_line(aes(y=diff)) +
  # theme_bw() +
  # geom_line(aes(y=min), color="red") +
  # geom_line(aes(y=max), color="blue") +
  # geom_smooth(se=FALSE) +
  ggtitle("Max Differene in Macroeconomic Negativity By Month") +
  xlab("Date") +
  ylab("Max Difference Negativity")
