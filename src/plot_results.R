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
  mutate(spellcheck_response = spellcheck_response * 100)
  calc_ner_f1("correctness_response", "informativeness_gold",
              "informativeness_response", "knowledge_response",
              "faithfulness_response")



llm_eval_scores <- d %>%
  select(-starts_with("informativeness_"), -starts_with("faithfulness_"),
         -starts_with("correctness_"), -starts_with("knowledge")) %>%
  pivot_longer(!c(model, id)) %>%
  #filter(name == "correctness") %>%
  filter(!is.na(value)) %>%
  mutate(is_llm = ifelse(str_detect(name, "response"),
                         "Automatic Readability",
                         "LLM Eval")) %>%
  mutate(name = str_split(name, "_") %>% map(pluck, 1) %>% unlist(9))


countbased <- d %>%
  select(model, id,
         starts_with("informativeness_"), starts_with("faithfulness_"),
         starts_with("correctness_"), starts_with("knowledge")) %>%
  select(model, id,
         ends_with("BERTscore_f1"), ends_with("rouge_l"),
         ends_with("rouge_1"), ends_with("ner_f1")) %>%
  pivot_longer(!c(model, id)) %>%
  filter(!is.na(value)) %>%
  mutate(origin = str_detect(name, "gold"),
         model = ifelse(origin, "human-gold-standard", model))


## histogram of
## ## llm eval scores
## llm_eval_scores %>%
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
  stat_ecdf(aes(value, color = model, linetype = fct_rev(is_llm))) +
  #scale_x_continuous(breaks = c(0.5, seq(0,10))) +
  facet_wrap(~name, scales="free") +
  theme_bw() +
  scale_color_viridis_d(begin=1/3) +
  theme(axis.text.y = element_blank(),
        axis.ticks.y = element_blank(),
        legend.position = "bottom",
        legend.box = "vertical",
        panel.grid.minor.x = element_blank(),
        text = element_text(size = 16),
        axis.text = element_text(size = 10)) +
  labs(fill = "Source", color = "Source", linetype = "Type",
       x = "", y = "")

ggsave("plots/results_lexical_semantic.png")

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
  facet_grid(goal ~ score, scale="free_y") +
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
