library(tidyverse)
library(rlang)

calc_ner_f1 <- function(.data, ...) {
  dots <- list(...)
  map_dfc(dots, function(arg) {
    recall <- .data[, str_c(arg, "_ner_recall")][[1]]
    precision <- .data[, str_c(arg, "_ner_precision")][[1]]
    x <- data_frame(2 * precision * recall / (pmax(precision + recall, 1)))
    names(x) <- str_c(arg, "_ner_f1")
    return(x)
  }) %>%
    bind_cols(.data) %>%
    return()
}


d <- read_csv("results/merged_eval_results.csv") %>%
  # rescale to 0-10 scale
  #sample_n(10) %>%
  #mutate(lix_response = lix_response / max(lix_response) * 10) %>%
  mutate(spellcheck_response = spellcheck_response * 100,
         spellcheck_gold = spellcheck_gold * 100) %>%
  calc_ner_f1("correctness_response", "informativeness_gold",
              "informativeness_response", "knowledge_response",
              "faithfulness_response")


## d %>%
##   filter(model == "mixtral-rag") %>%
##   summarize(lix = mean(lix_gold, na.rm=T), k_s = sd(lix_gold, na.rm=T),
##             f_m = mean(spellcheck_gold, na.rm=T), f_s = sd(spellcheck_gold, na.rm=T))


## d %>%
##   group_by(model) %>%
##   summarise(fa_rouge1_mean = mean(knowledge_response_rouge_1_precision, na.rm = T),
##             fa_rouge1_sd = sd(knowledge_response_rouge_1_precision, na.rm = T),
##             fa_rougel_mean = mean(knowledge_response_rouge_l_precision, na.rm = T),
##             fa_rougel_sd = sd(knowledge_response_rouge_l_precision, na.rm = T))


## filter(d, model == "mixtral-rag", id == "392") %>%
##   select(-model) %>%
##   select(starts_with("knowledge")) %>%
##   select(ends_with("knowledge"), ends_with("rouge_1_precision"), ends_with("rouge_l_precision"), ends_with("BERTscore_f1"), ends_with("ner_f1")) %>%# %>%
##   unlist() %>%
##   round(2)


## filter(d, id == 166) %>%
##   select(model, starts_with("knowledge"))%>%
##   select(model, ends_with("informativeness"), ends_with("correctness"), ends_with("knowledge"), ends_with("rouge_1_precision"), ends_with("rouge_l_precision"), ends_with("BERTscore_f1")) %>%
##   select(-starts_with("informativeness_gold")) %>%
##   print(width=500)


llm_eval_scores <- d %>%
  select(-starts_with("informativeness_"), -starts_with("faithfulness_"),
         -starts_with("correctness_"), -starts_with("knowledge_")) %>%
  pivot_longer(!c(model, id)) %>%
  #filter(name == "correctness") %>%
  filter(!is.na(value)) %>%
  mutate(is_llm = ifelse(str_detect(name, "response") | str_detect(name, "gold"),
                         "Automatic Readability",
                         "LLM Eval")) %>%
  mutate(origin = str_detect(name, "gold"),
         model = ifelse(origin, "human-gold-standard", model)) %>%
  mutate(name = str_split(name, "_") %>% map(pluck, 1) %>% unlist())


countbased <- d %>%
  select(model, id,
         starts_with("informativeness_"), starts_with("faithfulness_"),
         starts_with("correctness_"), starts_with("knowledge")) %>%
  mutate(knowledge_response_rouge_1 = knowledge_response_rouge_1_precision,
         knowledge_response_rouge_l = knowledge_response_rouge_l_precision,
         faithfulness_response_rouge_1 = faithfulness_response_rouge_1_precision,
         faithfulness_response_rouge_l = faithfulness_response_rouge_l_precision) %>%
  select(model, id,
         ends_with("BERTscore_f1"), ends_with("rouge_l"),
         ends_with("rouge_1"), ends_with("ner_f1")) %>%
  pivot_longer(!c(model, id)) %>%
  filter(!is.na(value)) %>%
  mutate(origin = str_detect(name, "gold"),
         model = ifelse(origin, "human-gold-standard", model))


## countbased %>%
##   filter(str_detect(name, "ner_f1")) %>%
##   group_by(model, name) %>%
##   summarise(mean = mean(value),
##             sd = sd(value))

## countbased %>%
##   filter(str_detect(name, "ner_f1")) %>%
##   filter(value > 0) %>%
##   group_by(model, name) %>%
##   summarise(mean = mean(value),
##             sd = sd(value))

## histogram of
## llm eval scores
## llm_eval_scores %>%
##   filter(name != "spellcheck", name != "lix") %>%
##   #group_by(model) %>%
##   #filter(id < 10) %>%
##   #summarise(across(!id, \(x) mean_se(na.exclude(x)))) %>%
##   #select(-document) %>%
##   #filter(model != "mixtral-simple-prompt") %>%
##   #filter(name == "correctness") %>%
##   ggplot() +
##   #geom_line(aes(factor(name), y=value, color=model, group=model), stat="summary", fun.y=mean) #+
##   #geom_density_ridges(aes(x = value, y = factor(name), group = str_c(name, model), color = model), fill = NA, stat="binline", bins = 11, scale = 0.95)
##   geom_histogram(aes(x=value, fill=model, group = str_c(name, model)), color=NA, position=position_dodge(width=0.7), bins=11) +
##   #stat_count(aes(x = value, fill=model)) +
##   #geom_line(aes(x = factor(model), y=value, group=id), alpha=.15) + #+
##   #geom_pointrange(stat="summary", fun.data=mean_cl_normal, alpha=0.2, color=NA)+
##   #coord_cartesian(ylim = c(0,10))  +
##   #scale_x_continuous(breaks = c(seq(0,10), 20, 40, 60, 80)) +
##   facet_grid(name ~. )

## ecdf
llm_eval_scores %>%
  ggplot() +
  stat_ecdf(aes(value, color = model, linetype = fct_rev(is_llm)), size = 1.5) +
  #scale_x_continuous(breaks = c(0.5, seq(0,10))) +
  facet_wrap(~name, scales="free") +
  theme_bw() +
  scale_color_viridis_d() +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.position = "bottom",
        legend.box = "vertical",
        panel.grid.minor.x = element_blank(),
        text = element_text(size = 16),
        axis.text = element_text(size = 10)) +
  labs(fill = "Source", color = "Source", linetype = "Type",
       x = "", y = "")

ggsave("plots/results_llmeval_autoread.png")

## ## histogram of
## ## eval scores
## d %>%
##   mutate(lix_response = lix_response / max(lix_response) * 10) %>%
##   group_by(model) %>%
##   #filter(id < 10) %>%
##   #summarise(across(!id, \(x) mean_se(na.exclude(x)))) %>%
##   select(-starts_with("informativeness_"), -starts_with("faithfulness_"), -starts_with("correctness_")) %>%
##   mutate(spellcheck_response = spellcheck_response * 10) %>%
##   select(-document) %>%
##   #filter(model != "mixtral-simple-prompt") %>%
##   pivot_longer(!c(model, id)) %>%
##   #filter(name == "correctness") %>%
##   filter(!is.na(value)) %>%
##   ggplot() +
##   #geom_line(aes(factor(name), y=value, color=model, group=model), stat="summary", fun.y=mean) #+
##   #geom_density_ridges(aes(x = value, y = factor(name), group = str_c(name, model), color = model), fill = NA, stat="binline", bins = 11, scale = 0.95)
##   geom_histogram(aes(x=value, fill=model, group = str_c(name, model)), color=NA, position=position_dodge(width=0.7), bins=11) +
##   #stat_count(aes(x = value, fill=model)) +
##   #geom_line(aes(x = factor(model), y=value, group=id), alpha=.15) + #+
##   #geom_pointrange(stat="summary", fun.data=mean_cl_normal, alpha=0.2, color=NA)+
##   #coord_cartesian(ylim = c(0,10))  +
##   scale_x_continuous(breaks = seq(0,10)) +
##   facet_grid(name ~. )


######
## plot count based
countbased %>%
  arrange(model, name) %>%
  separate_wider_regex(name, c(goal = "[^_]+", "_[^_]+_", score = ".+")) %>%
  mutate(goal = fct_relevel(goal, "informativeness")) %>%
  ggplot() +
  geom_density(aes(value, color = model), size=1.5) +
  #geom_histogram(aes(value, fill=model), bins = 11, position=position_dodge(width=0.9))+
    #stat_ecdf(aes(value, color = model)) +
  facet_grid(score ~ goal, scales = "free_y") +
  theme_bw() +
  scale_color_viridis_d() +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        text = element_text(size = 16),
        axis.text = element_text(size = 10)) +
  labs(fill = "Source", color = "Source",
       x = "", y = "") +
  theme(legend.position = "bottom")

ggsave("plots/results_lexical_semantic.png")
