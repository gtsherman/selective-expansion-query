# Create a "flex" oracle with multiple acceptable classes.
flex = function(scores, threshold=0.02) {
  scores %>% 
    filter(index != 'baseline') %>%
    group_by(query) %>% 
    mutate(good = ifelse(map >= max(map) - threshold, 1, 0)) %>% 
    ungroup() 
}

# Convert features to a condensed form where the values for "good" and "bad"
# runs are shown side by side for each feature/query pair.
condense = function(df, df_oracle_flex) {
  df %>% 
    gather(jaccard, expansion_rm, target_rm, 
           doc_vs_expansion, pairwise, doc_clarity,
           exp_doc_clarity, key = 'metric', value = 'value') %>% 
    merge(df_oracle_flex) %>% 
    select(-map) %>% 
    group_by(query, good, metric) %>% 
    summarize(value = mean(value)) %>% 
    spread(good, value) %>%
    rename(good = `0`, bad = `1`)
}

condense_qpp = function(df, df_oracle_flex) {
  df %>% 
    gather(clarity, wig, nqc,
           key = 'metric', value = 'value') %>% 
    merge(df_oracle_flex) %>% 
    select(-map) %>% 
    group_by(query, good, metric) %>% 
    summarize(value = mean(value)) %>% 
    spread(good, value) %>%
    rename(good = `0`, bad = `1`)
}

# Further condense features to show only the difference between the values for
# "good" and "bad" runs.
difference = function(condensed) {
  condensed %>% 
    mutate(diff = good - bad) %>% 
    filter(!is.na(diff)) %>% 
    select(query, metric, diff) %>% 
    ungroup()
}

# An example plot showing how these features don't differ
difference(condense(ap_qpp, ap_oracle_flex)) %>% 
  ggplot(aes(x = diff)) + 
    geom_histogram(binwidth = 0.03) + 
    geom_vline(xintercept = 0.0, color = 'red') + 
    facet_grid(metric ~ .)

# T-tests showing that the good and bad features are
# very solidly not different.
condense(ap, ap_oracle_flex) %>%
  group_by(metric) %>%
  summarize(t.test(good, bad)$p.value)

# Similar approach, but where only the best and worst runs are compared. Does this by limiting
# the flex oracle to only include the max and min map values.
condense(ap,
         flex(ap_scores %>%
                group_by(query) %>%
                filter(map == max(map) | map == min(map)))) %>%
  group_by(metric) %>%
  summarize(t.test(good, bad)$p.value)
