###############################################################################
# BANK LOAN APPROVAL - PROGETTO R
# Autrici: Cecilia Morici, Francesca Selaj, Morena Farinelli
# Obiettivo: prevedere la concessione di un prestito personale usando modelli
# supervisionati: Regressione Logistica, Random Forest e Albero Decisionale.
#
# NOTA IMPORTANTE:
# Questo script è pensato per essere usato in RStudio.
# Salva questo file nella stessa cartella del dataset bank.csv, oppure metti il
# dataset nella cartella data/bank.csv.
###############################################################################

rm(list = ls())

# =============================================================================
# 1. LIBRERIE
# =============================================================================

required_packages <- c(
  "rstudioapi",
  "readr",
  "dplyr",
  "tidyr",
  "ggplot2",
  "caret",
  "caTools",
  "pROC",
  "ranger",
  "ggcorrplot",
  "rpart",
  "rpart.plot"
)

missing_packages <- required_packages[!required_packages %in% rownames(installed.packages())]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Mancano questi pacchetti: ",
      paste(missing_packages, collapse = ", "),
      "\nInstallali con: install.packages(c('",
      paste(missing_packages, collapse = "', '"),
      "'))"
    )
  )
}

invisible(lapply(required_packages, library, character.only = TRUE))

# =============================================================================
# 2. IMPOSTAZIONE CARTELLA DI LAVORO - SOLO RSTUDIO
# =============================================================================

if (!rstudioapi::isAvailable()) {
  stop("Questo script è pensato per essere eseguito da RStudio.")
}

current_path <- rstudioapi::getActiveDocumentContext()$path

if (is.null(current_path) || current_path == "") {
  stop("Prima salva questo file .R, poi eseguilo da RStudio.")
}

project_dir <- dirname(current_path)
setwd(project_dir)

cat("Cartella di lavoro impostata su:\n", getwd(), "\n\n")

# =============================================================================
# 3. LETTURA DATASET
# =============================================================================

possible_data_paths <- c("bank.csv", file.path("data", "bank.csv"))
data_path <- possible_data_paths[file.exists(possible_data_paths)][1]

if (is.na(data_path)) {
  stop(
    "File bank.csv non trovato. Metti bank.csv nella stessa cartella dello script, ",
    "oppure nella cartella data/bank.csv."
  )
}

data_raw <- readr::read_csv(data_path, show_col_types = FALSE)

cat("Dataset caricato da:", data_path, "\n")
cat("Dimensioni dataset originale:", nrow(data_raw), "righe e", ncol(data_raw), "colonne\n\n")

# Se vuoi visualizzare il dataset in RStudio, togli il commento alla riga sotto.
# View(data_raw)

# =============================================================================
# 4. DATA CLEANING
# =============================================================================

# Rimuoviamo ID e ZIP.Code perché sono identificativi e non aggiungono
# informazione utile al modello predittivo.
data <- data_raw %>%
  dplyr::select(-dplyr::any_of(c("ID", "ZIP.Code")))

# Correggiamo valori negativi di Experience: l'esperienza lavorativa non può
# essere negativa, quindi sostituiamo quei valori con 0.
data <- data %>%
  dplyr::mutate(
    Experience = dplyr::if_else(Experience < 0, 0, Experience)
  )

cat("Struttura dataset dopo cleaning:\n")
str(data)

cat("\nValori mancanti totali:", sum(is.na(data)), "\n")
cat("Righe duplicate:", sum(duplicated(data)), "\n\n")

# Nota: le righe duplicate vengono solo segnalate, non eliminate, perché nel
# dataset bancario due clienti possono avere caratteristiche uguali.

# =============================================================================
# 5. ANALISI ESPLORATIVA
# =============================================================================

cat("Distribuzione Personal.Loan:\n")
print(table(data$Personal.Loan))
cat("\nDistribuzione percentuale Personal.Loan:\n")
print(round(prop.table(table(data$Personal.Loan)), 4))
cat("\nMedia Personal.Loan, cioè quota di clienti con prestito:",
    round(mean(data$Personal.Loan), 4), "\n\n")

# Grafico a torta della variabile target
pie(
  table(data$Personal.Loan),
  main = "Distribuzione del Prestito",
  labels = c("NO", "YES"),
  col = c("lightpink", "lightgreen")
)

# Grafico a barre della variabile target
loan_counts <- table(data$Personal.Loan)
barplot_heights <- barplot(
  loan_counts,
  main = "Distribuzione del Prestito",
  col = c("red", "lightgreen"),
  names.arg = c("NO", "YES"),
  ylab = "Frequenza",
  ylim = c(0, max(loan_counts) * 1.15)
)

text(
  x = barplot_heights,
  y = loan_counts / 2,
  labels = c("NO", "YES"),
  col = "black",
  cex = 1.1
)

text(
  x = barplot_heights,
  y = loan_counts + max(loan_counts) * 0.03,
  labels = as.character(loan_counts),
  col = "black",
  cex = 1
)

# Subset dei clienti a cui è stato concesso il prestito
# Qui Personal.Loan è ancora numerica: 0 = NO, 1 = YES.
data_yes <- subset(data, Personal.Loan == 1)
cat("\nSummary dei clienti con prestito concesso:\n")
print(summary(data_yes))

# Boxplot delle principali variabili numeriche
numeric_vars <- c("Age", "Experience", "Income", "Family", "Mortgage", "CCAvg", "Education")
boxplot_colors <- c("salmon", "lightcoral", "lightgreen", "lightyellow",
                    "lightblue", "peachpuff", "lightsteelblue")

for (i in seq_along(numeric_vars)) {
  boxplot(
    data[[numeric_vars[i]]],
    col = boxplot_colors[i],
    main = numeric_vars[i],
    ylab = numeric_vars[i]
  )
}

# Grafici per variabili dicotomiche
binary_vars <- c("Personal.Loan", "Securities.Account", "CD.Account", "Online", "CreditCard")

for (var in binary_vars) {
  counts <- table(data[[var]])
  barplot(
    counts,
    main = paste("Distribuzione", var),
    col = c("lightblue", "lightgreen"),
    xlab = var,
    ylab = "Frequenza"
  )
}

# Matrice di correlazione su variabili numeriche
corr_matrix <- cor(data, use = "complete.obs")
corr_plot <- ggcorrplot::ggcorrplot(
  corr_matrix,
  type = "lower",
  lab = FALSE,
  title = "Matrice di correlazione"
)
print(corr_plot)

# =============================================================================
# 6. PREPARAZIONE DATI PER I MODELLI
# =============================================================================

# Per caret la variabile target deve essere un fattore con nomi validi.
data_model <- data %>%
  dplyr::mutate(
    Personal.Loan = factor(Personal.Loan, levels = c(0, 1), labels = c("NO", "YES"))
  )

set.seed(123)
split <- caTools::sample.split(data_model$Personal.Loan, SplitRatio = 0.75)
train_data <- subset(data_model, split == TRUE)
test_data  <- subset(data_model, split == FALSE)

cat("\nDimensioni training set:\n")
print(dim(train_data))
cat("Dimensioni test set:\n")
print(dim(test_data))

cat("\nBilanciamento training set:\n")
print(round(prop.table(table(train_data$Personal.Loan)), 4))
cat("\nBilanciamento test set:\n")
print(round(prop.table(table(test_data$Personal.Loan)), 4))

# Funzione personalizzata per la cross-validation.
# In questo modo la ROC, la sensibilità e la specificità vengono calcolate
# considerando YES come classe positiva, cioè il caso più importante per il progetto.
loan_summary <- function(data, lev = NULL, model = NULL) {
  roc_obj <- pROC::roc(
    response = data$obs,
    predictor = data[, "YES"],
    levels = c("NO", "YES"),
    direction = "<",
    quiet = TRUE
  )

  c(
    ROC = as.numeric(pROC::auc(roc_obj)),
    Sens = caret::sensitivity(data$pred, data$obs, positive = "YES"),
    Spec = caret::specificity(data$pred, data$obs, negative = "NO")
  )
}

# Controllo per cross-validation con upsampling della classe minoritaria.
ctrl <- caret::trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = loan_summary,
  savePredictions = "final",
  sampling = "up",
  verboseIter = TRUE
)

# Funzione per estrarre le metriche principali dalla confusion matrix.
extract_metrics <- function(model_name, confusion_matrix, auc_value) {
  data.frame(
    Modello = model_name,
    Accuracy = as.numeric(confusion_matrix$overall["Accuracy"]),
    Sensitivity = as.numeric(confusion_matrix$byClass["Sensitivity"]),
    Specificity = as.numeric(confusion_matrix$byClass["Specificity"]),
    Precision = as.numeric(confusion_matrix$byClass["Pos Pred Value"]),
    Balanced_Accuracy = as.numeric(confusion_matrix$byClass["Balanced Accuracy"]),
    AUC = as.numeric(auc_value)
  )
}

# =============================================================================
# 7. MODELLO 1 - REGRESSIONE LOGISTICA
# =============================================================================

set.seed(123)
model_logit <- caret::train(
  Personal.Loan ~ .,
  data = train_data,
  method = "glm",
  family = binomial,
  metric = "ROC",
  trControl = ctrl,
  preProcess = c("center", "scale")
)

cat("\n================ REGRESSIONE LOGISTICA ================\n")
print(model_logit)
print(summary(model_logit$finalModel))
cat("AIC modello logit:", AIC(model_logit$finalModel), "\n")

# Predizioni probabilistiche sul test set
pred_prob_logit <- predict(model_logit, newdata = test_data, type = "prob")[, "YES"]

# Soglia scelta: 0.30, utile in presenza di classi sbilanciate per aumentare
# la capacità del modello di intercettare i casi YES.
threshold_logit <- 0.30
pred_class_logit <- ifelse(pred_prob_logit >= threshold_logit, "YES", "NO")
pred_class_logit <- factor(pred_class_logit, levels = c("NO", "YES"))

cm_logit <- caret::confusionMatrix(
  data = pred_class_logit,
  reference = test_data$Personal.Loan,
  positive = "YES"
)

cat("\nConfusion Matrix - Regressione Logistica:\n")
print(cm_logit)

roc_logit <- pROC::roc(
  response = test_data$Personal.Loan,
  predictor = pred_prob_logit,
  levels = c("NO", "YES"),
  direction = "<"
)
auc_logit <- pROC::auc(roc_logit)
cat("AUC Regressione Logistica:", auc_logit, "\n")

plot(roc_logit, col = "red", main = "Curva ROC - Regressione Logistica")

# =============================================================================
# 8. MODELLO 2 - RANDOM FOREST
# =============================================================================

rf_grid <- expand.grid(
  mtry = c(2, 4, 6),
  splitrule = "gini",
  min.node.size = c(1, 5)
)

set.seed(123)
model_rf <- caret::train(
  Personal.Loan ~ .,
  data = train_data,
  method = "ranger",
  metric = "ROC",
  trControl = ctrl,
  preProcess = c("center", "scale"),
  tuneGrid = rf_grid,
  num.trees = 500,
  importance = "impurity"
)

cat("\n================ RANDOM FOREST ================\n")
print(model_rf)

pred_prob_rf <- predict(model_rf, newdata = test_data, type = "prob")[, "YES"]
pred_class_rf <- predict(model_rf, newdata = test_data)

cm_rf <- caret::confusionMatrix(
  data = pred_class_rf,
  reference = test_data$Personal.Loan,
  positive = "YES"
)

cat("\nConfusion Matrix - Random Forest:\n")
print(cm_rf)

roc_rf <- pROC::roc(
  response = test_data$Personal.Loan,
  predictor = pred_prob_rf,
  levels = c("NO", "YES"),
  direction = "<"
)
auc_rf <- pROC::auc(roc_rf)
cat("AUC Random Forest:", auc_rf, "\n")

plot(roc_rf, col = "darkgreen", main = "Curva ROC - Random Forest")

# Importanza variabili Random Forest
var_imp_rf <- caret::varImp(model_rf)
print(var_imp_rf)
print(plot(var_imp_rf, main = "Importanza delle variabili - Random Forest"))

# =============================================================================
# 9. MODELLO 3 - ALBERO DECISIONALE
# =============================================================================

# Per l'albero decisionale trattiamo come fattori le variabili categoriche.
factor_predictors <- c("CreditCard", "Online", "Securities.Account", "CD.Account", "Education")

train_tree <- train_data %>%
  dplyr::mutate(dplyr::across(dplyr::all_of(factor_predictors), as.factor))

test_tree <- test_data %>%
  dplyr::mutate(dplyr::across(dplyr::all_of(factor_predictors), as.factor))

set.seed(123)
tree_full <- rpart::rpart(
  Personal.Loan ~ .,
  data = train_tree,
  method = "class",
  control = rpart::rpart.control(cp = 0.0001),
  parms = list(split = "gini")
)

cat("\n================ ALBERO DECISIONALE ================\n")
print(tree_full)
cat("\nTabella CP albero completo:\n")
printcp(tree_full)

plotcp(tree_full, main = "Scelta del parametro CP")

best_cp <- tree_full$cptable[which.min(tree_full$cptable[, "xerror"]), "CP"]
cat("Miglior CP selezionato:", best_cp, "\n")

tree_pruned <- rpart::prune(tree_full, cp = best_cp)

cat("\nAlbero dopo pruning:\n")
print(tree_pruned)
print(summary(tree_pruned))

rpart.plot::rpart.plot(
  tree_pruned,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  faclen = 0,
  cex = 0.7,
  main = "Albero Decisionale Pruned"
)

pred_class_tree <- predict(tree_pruned, newdata = test_tree, type = "class")
pred_prob_tree <- predict(tree_pruned, newdata = test_tree, type = "prob")[, "YES"]

cm_tree <- caret::confusionMatrix(
  data = pred_class_tree,
  reference = test_tree$Personal.Loan,
  positive = "YES"
)

cat("\nConfusion Matrix - Albero Decisionale:\n")
print(cm_tree)

roc_tree <- pROC::roc(
  response = test_tree$Personal.Loan,
  predictor = pred_prob_tree,
  levels = c("NO", "YES"),
  direction = "<"
)
auc_tree <- pROC::auc(roc_tree)
cat("AUC Albero Decisionale:", auc_tree, "\n")

plot(roc_tree, col = "blue", main = "Curva ROC - Albero Decisionale")
abline(a = 0, b = 1, lty = 2, col = "gray")

# =============================================================================
# 10. CONFRONTO TRA I MODELLI
# =============================================================================

# Confronto ROC nello stesso grafico
plot(roc_logit, col = "red", main = "Confronto Curve ROC")
lines(roc_rf, col = "darkgreen")
lines(roc_tree, col = "blue")
abline(a = 0, b = 1, lty = 2, col = "gray")
legend(
  "bottomright",
  legend = c(
    paste0("Logit - AUC: ", round(auc_logit, 4)),
    paste0("Random Forest - AUC: ", round(auc_rf, 4)),
    paste0("Decision Tree - AUC: ", round(auc_tree, 4))
  ),
  col = c("red", "darkgreen", "blue"),
  lwd = 2,
  cex = 0.8
)

# Tabella metriche finali calcolate direttamente dai modelli.
model_metrics <- dplyr::bind_rows(
  extract_metrics("Regressione Logistica", cm_logit, auc_logit),
  extract_metrics("Random Forest", cm_rf, auc_rf),
  extract_metrics("Albero Decisionale", cm_tree, auc_tree)
)

cat("\n================ CONFRONTO FINALE MODELLI ================\n")
model_metrics_print <- model_metrics %>%
  dplyr::mutate(dplyr::across(where(is.numeric), ~ round(.x, 4)))
print(model_metrics_print)

metrics_long <- model_metrics %>%
  dplyr::select(Modello, Accuracy, Sensitivity, Specificity, Precision, AUC) %>%
  tidyr::pivot_longer(
    cols = -Modello,
    names_to = "Metrica",
    values_to = "Valore"
  )

comparison_plot <- ggplot(metrics_long, aes(x = Modello, y = Valore, fill = Metrica)) +
  geom_col(position = "dodge") +
  ylim(0, 1) +
  labs(
    title = "Confronto Modelli: Metriche di Performance",
    x = "Modello",
    y = "Valore"
  ) +
  theme_minimal()

print(comparison_plot)

# =============================================================================
# 11. SALVATAGGIO RISULTATI
# =============================================================================

# I risultati vengono salvati in una cartella outputs, utile per GitHub.
if (!dir.exists("outputs")) {
  dir.create("outputs")
}

readr::write_csv(model_metrics, file.path("outputs", "model_metrics.csv"))
saveRDS(model_logit, file.path("outputs", "model_logit.rds"))
saveRDS(model_rf, file.path("outputs", "model_random_forest.rds"))
saveRDS(tree_pruned, file.path("outputs", "model_decision_tree.rds"))

cat("\nScript completato correttamente. Risultati salvati nella cartella outputs/.\n")

###############################################################################
# FINE SCRIPT
###############################################################################
