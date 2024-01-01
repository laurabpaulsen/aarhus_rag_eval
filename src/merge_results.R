library(tidyverse)
library(tidyjson)



llm_docs <- read_csv("results/mixtral-rag_llm_documents.csv") %>%
  mutate(i = 1:n(),
         model = "mixtral-rag") %>%
  select(model, i, faithfulness, knowledge)

llm_results <- list.files("results", pattern="*llm.csv", full.names = TRUE) %>%
  map(~read_csv(.x) %>% mutate(model = str_extract(.x, "results/(.*)_llm.csv", group=1))) %>%
  map(mutate, i = 1:n()) %>%
  bind_rows() %>%
  rename(gold = reference,
         model_response = proposed) %>%
  left_join(llm_docs)



countbased_results <- list.files("results/merged/", full.names = TRUE) %>%
  map(function(.x) {
    d <- read_csv(.x)
    metadata <- str_match(.x, "results/merged//(.*?)_(.*)\\.csv")
    d$model = metadata[,2]
    d$comparison = metadata[,3]
    return(d)
  }) %>%
  bind_rows() %>%
  mutate(name = case_when(comparison == "gold_to_response" ~ "correctness_response",
                          comparison == "question_to_gold" ~ "informativeness_gold",
                          comparison == "question_to_response" ~ "informativeness_response",
                          comparison == "document_to_question" ~ "knowledge_response",
                          comparison == "document_to_response" ~ "faithfulness_response")) %>%
  #filter(name != "faithfulness_gold") %>% ## this one didnt make sense after all
  separate_wider_delim(comparison, delim = "_to_", names = c("reference_name", "candidate_name"))


x <- countbased_results %>%
  mutate(question = case_when(reference_name == "question" ~ reference,
                              candidate_name == "question" ~ candidate),
         gold = case_when(reference_name == "gold" ~ reference,
                          candidate_name == "gold" ~ candidate),
         model_response = case_when(reference_name == "response" ~ reference,
                                    candidate_name == "response" ~ candidate),
         document = case_when(reference_name == "document" ~ reference,
                              candidate_name == "document" ~ candidate))##  %>%
  ## sample_n(10)


x2 <- pivot_longer(x, cols = c(starts_with("ner"), starts_with("rouge"), starts_with("BERT")), names_to="measure", values_to = "value") %>%
  mutate(measure = str_c(name, measure, sep="_")) %>%
  select(-reference_name, -candidate_name, -name, -reference, -candidate) %>%
  pivot_wider(names_from = measure,
              values_from = value)


## x3 <- x2 %>%
##   dplyr::group_by(model, question, gold, model_response, document, measure) %>%
##   dplyr::summarise(n = dplyr::n(), .groups = "drop") %>%
##   dplyr::filter(n > 1L)


merged_results <- llm_results %>%
  select(-documents) %>%
  left_join(select(x2, model, gold, model_response, starts_with("correctness_response")), by = join_by(model, gold, model_response), keep=FALSE, na_matches='never') %>%
  left_join(select(x2, question, gold, starts_with("informativeness_gold")), by = join_by(question, gold), keep=FALSE, na_matches='never') %>%
  left_join(select(x2, model, question, model_response, starts_with("informativeness_response")), by = join_by(model, question, model_response), keep=FALSE, na_matches='never') %>%
  left_join(select(x2, model, document, question, starts_with("knowledge_response")) %>% filter(model == "mixtral-rag", complete.cases(.)), by = join_by(model, question), keep=FALSE, na_matches='never') %>%
  left_join(select(x2, model, model_response, starts_with("faithfulness_response")) %>% filter(model == "mixtral-rag", complete.cases(faithfulness_response_rouge_l)) %>% distinct(), by = join_by(model, model_response), keep=FALSE, na_matches='never')# %>%
# left_join(select(x2, model, gold, starts_with("faithfulness_gold")), by = join_by(model, gold), keep=FALSE, na_matches='never') %>%

  #pivot_wider(-model, id_cols = model)



readability_results <-
  read_file("results/readability.json") %>%
  spread_all() %>%
  gather_object("model") %>%
  gather_object("name") %>%
  #enter_object("lix_answer") %>%
  gather_array() %>%
  spread_values(value = jdouble()) %>%
  as_tibble() %>%
  pivot_wider(names_from = "name", values_from = value) %>%
  select(-document.id) %>%
  rename(i = array.index, lix_response = lix_answer, spellcheck_response = spellcheck_answer)

final_results <- left_join(merged_results, readability_results) %>%
  rename(id = i)


# save a "record of results"
select(final_results, model, id, !c(question, gold, model_response, document))%>%
  write_csv( "results/merged_eval_results.csv")

# also save something for us to look at
final_results %>%
  write_csv("results/merged_eval_results_FULL.csv")
