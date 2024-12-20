library(jsonlite)

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
         str_detect(source, "nytimes|washington|wsj")) %>% 
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
  ggtitle("% of Macroeconomic Articles That Were Negative (Per Month)")

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
         str_detect(source, "wsj")) %>% 
  mutate(year = year(date),
         month = month(date)) %>% 
  group_by(year, month, macro_type) %>% 
  summarise(count = n()) %>% 
  group_by(year, month) %>% 
  mutate(tot = sum(count),
         per = count / tot,
         date = as_date(paste(year, month, "1", sep="-"))) %>% 
  ungroup() %>%
  filter(year > 2014, year != 2021,
         str_detect(macro_type,"job|prices|macro")) %>% 
  ggplot(aes(x=date, y=per)) +
  geom_line(aes(color=macro_type, group=macro_type), size=1, alpha=0.7) +
  ggtitle("Macro Indicator Use (WSJ)")

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
  ggtitle("% of articles with inflation datapoints")

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

