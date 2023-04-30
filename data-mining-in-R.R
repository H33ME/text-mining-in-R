# set the working directory
setwd('./Text-Mining')

# load the packages
library(tidyverse)
library(tidytext)
library(textdata)
library(gridExtra)
library(quanteda)
library(tm)
library(topicmodels)
library(wordcloud)
# load the data
state_data <- read_csv("./state_of_the_union.csv")

# select the text
george_speech <- state_data %>%
  filter(president == 'George Washington')

text <- george_speech$text
# clean the text
# Remove newline characters and panctuation and convert to lowercases
clean_text <- tolower(gsub("[\n\r[:punct:]]", " ", text))

text_df <- tibble(
  line = 1:length(text),
  text = text,
  speech_doc_id = george_speech$speech_doc_id
)
#break the text into individual tokens ie tokenisation
#use unnest_tokens() from tidytext package
tidy_text <- text_df %>%
  unnest_tokens(word, text) %>%
  filter(!grepl('[0-9]', word))

#remove the stop_words with anti_join()
data("stop_words")
tidy_text <- tidy_text %>%
  anti_join(stop_words)

# afinn and bing sentiment dictionary
afinn <- get_sentiments('afinn')
bing <- get_sentiments('bing')
# apply both to data
sentiments <- tidy_text %>%
  inner_join(afinn) %>%
  inner_join(bing) %>%
  mutate(
    sentiment = case_when(
      value > 0 & sentiment == 'positive' ~ 'positive',
      value < 0 & sentiment == 'negative' ~ 'negative',
      TRUE ~ 'neutral'
    )
  )
# positive sentiment
positive_sentiment <- sentiments %>%
  filter(sentiment == 'positive') %>%
  group_by(speech_doc_id) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
# negative sentiment
negative_sentiment <- sentiments %>%
  filter(sentiment == 'negative') %>%
  group_by(speech_doc_id) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
# positive sentiment barplot
pos_barplot <- positive_sentiment %>%
  ggplot(aes(x = speech_doc_id, y = count, fill = speech_doc_id)) +
  geom_bar(stat = 'identity') +
  coord_flip() +
  labs(title = "A bar plot showing most positive speech and which address it comes from",
       x = 'Speech Address', y = 'count')
# negative sentiment barplot
neg_barplot <- negative_sentiment %>%
  ggplot(aes(x = speech_doc_id, y = count, fill = speech_doc_id)) +
  geom_bar(stat = 'identity') +
  coord_flip() +
  labs(title = "A bar plot showing most negative speech and which address it comes from",
       x = 'Speech Address', y = 'count')
grid.arrange(pos_barplot, neg_barplot, ncol = 2)
#### Question 2
# Preprocess the text
text1 <- tolower(gsub("[\n\r[:punct:]]", " ", state_data$text))
text_df1 <-
  tibble(
    line = 1:length(text1),
    text = text1,
    president = state_data$president,
    date = state_data$date
  )

# Tokenize the text
tidy_text1 <- text_df1 %>%
  unnest_tokens(word, text) %>%
  filter(!grepl('[0-9]', word))
# stop words
data("stop_words")
tidy_text1 <- tidy_text1 %>%
  anti_join(stop_words)
# nrc for positive and negative sentiment
nrc_positive <- get_sentiments('nrc') %>%
  filter(sentiment == 'positive')

nrc_negative <- get_sentiments('nrc') %>%
  filter(sentiment == 'negative')
# get the positive and negative sentiments
positive_sentiments <- tidy_text1 %>%
  anti_join(nrc_positive) %>%
  group_by(president, date) %>%
  summarize(count = n()) %>%
  arrange(desc(count))

negative_sentiments <- tidy_text1 %>%
  anti_join(nrc_negative) %>%
  group_by(president, date) %>%
  summarize(count = n()) %>%
  arrange(desc(count))
# plot for president with positive and negative sentiment
#first 10
positive_sentiment_plot <- positive_sentiments %>%
  filter(count >= 5000) %>%
  ggplot(aes(x = date, y = count, fill = president)) +
  geom_col() +
  coord_flip() +
  labs(
    title = 'Plot for the positive sentiments from different president',
    subtitle = 'President with the most positive address is William McKinley and the date is 1898-12-05 with 6986 positive words',
    x = "Date",
    y = 'Count'
  )
negative_sentiment_plot <- negative_sentiments %>%
  filter(count >= 5000) %>%
  ggplot(aes(x = date, y = count, fill = president)) +
  geom_col() +
  coord_flip() +
  labs(
    title = 'Plot for the negative sentiments from different presidents in different time period',
    subtitle = 'President with the most negative address is William Mckinley and the date is 1898-12-05 with a count of 7593 negative words',
    x = 'Date',
    y = 'Count'
  )
grid.arrange(positive_sentiment_plot, negative_sentiment_plot, ncol = 2)
## Question 3
# custom stop words
words <-
  c(
    'government',
    'united',
    'states',
    'congress',
    'country',
    'american',
    'citizens',
    'public',
    'people'
  )
custom_stop_words <- bind_rows(tibble(word = words,
                                      lexicon = c(rep(
                                        "custom", times = length(words)
                                      ))),
                               stop_words)
# word counts as per the president speech
speech <- tolower(gsub("[\n\r[:punct:]]", " ", state_data$text))
speech_df <- tibble(
  line = 1:length(speech),
  text = speech,
  president = state_data$president
)
speech_df <- speech_df %>%
  unnest_tokens(word, text) %>%
  filter(!grepl('[0-9]', word)) %>%
  anti_join(custom_stop_words, by = 'word')
# create word count data frame
word_count <- speech_df %>%
  group_by(president) %>%
  count(word, president, sort = TRUE)

# view the resulting word count data frame
word_count
# term document matrix
tidy_text_df <- tibble(line = 1:length(speech), text = speech)
tidy_text_df <- tidy_text_df %>%
  unnest_tokens(word, text) %>%
  anti_join(stop_words) %>%
  filter(!grepl('[0-9]', word))

term_doc_matrix <- tidy_text_df$word %>%
  dfm()

# perform lda
k <- 5
# Fit the LDA model
lda <- LDA(term_doc_matrix, k, control = list(seed = 1337))

# word-topic probabilities
lda_topics <- tidy(lda, matrix = 'beta')
lda_top_terms <- lda_topics %>%
  group_by(topic) %>%
  slice_max(beta, n = 10) %>%
  ungroup() %>%
  arrange(topic, -beta)
# visualize top 10
lda_top_terms %>%
  mutate(term = reorder_within(term, beta, topic)) %>%
  ggplot(aes(beta, term, fill = factor(topic))) +
  geom_col(show.legend = F) +
  facet_wrap( ~ topic, scale = 'free') +
  scale_y_reordered()
# adding labels to the group
group_label <-
  
  ## question 4
  abraham_lincoln_data <- state_data %>%
  filter(president == 'Abraham Lincoln')
abraham_lincoln_speech <-
  tolower(gsub("[\n\r[:punct:]]", " ", abraham_lincoln_data$text))
abraham_lincoln_df <- tibble(line = 1:length(abraham_lincoln_speech),
                             text = abraham_lincoln_speech)
abraham_lincoln_df <- abraham_lincoln_df %>%
  unnest_tokens(word, text) %>%
  filter(!grepl('[0-9]', word)) %>%
  anti_join(stop_words)
abraham_lincoln_df %>%
  count(word, sort = TRUE) %>%
  slice_max(n, n = 500) %>%
  with(wordcloud(word, n))
