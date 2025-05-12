
install.packages(c("dplyr","tidyverse", "readr", "stringr", "tidyr", "mice", "DataExplorer", "caret", "ggplot2", "lattice")) 
library(dplyr) 
library(readr) 
library(stringr) 
library(tidyr) 
library(mice) 
library(DataExplorer) 
library(caret) 
library (ggplot2) 
library (lattice) 
library(tidyverse)
library(caTools)
library(glmnet)
library(pROC)


#read the data
df<- read_csv("C:/Users/Yousra/Desktop/healthcare-dataset-stroke-data_v1.csv")
str(df)
head(df)
summary(df)
View(df)

#Hnadle Missing values in 'bmi' with mean
plot_missing(df)
df$bmi = ifelse(is.na(df$bmi),
                ave(df$bmi, FUN = function(x) mean(x, na.rm = TRUE)),
                df$bmi)

# outliers handling
handle_outliers <- function(df, column) {  
  Q1 <- quantile(df[[column]], 0.25)   
  Q3 <- quantile(df[[column]], 0.75)   
  IQR <- Q3 - Q1   
  lower_bound <- Q1 - 1.5 * IQR  
  upper_bound <- Q3 + 1.5 * IQR  
  df[[column]] <- ifelse(df[[column]] > upper_bound, upper_bound,                         
                         ifelse(df[[column]] < lower_bound, lower_bound, df[[column]]))   
  return(df) }

# Apply the function to each numeric column in the data frame
df <- handle_outliers(df, 'bmi')
df <- handle_outliers(df, 'age')
df <- handle_outliers(df, 'hypertension')
df <- handle_outliers(df, 'avg_glucose_level')


# Plot boxplots for each numerical variable to visualize outliers 
numerical_colnames <- names(df)[sapply(df, is.numeric)] 
par(mfrow=c(2,2)) 
for (col in numerical_colnames) {  
  boxplot(df[[col]], main=col, col="green", border="black", horizontal=TRUE, outline=TRUE) 
}

#drop unnecessary column  
cols_to_remove <- c("id", "work_type", "Residence_type", "smoking_status")
df <- df %>% select(-all_of(cols_to_remove))

#Encoding categorical variables
df$gender <- factor(df$gender,
                    levels = c('Male', 'Female', 'Other'),
                    labels = c(1, 2, 3))

df$ever_married <- factor(df$ever_married,
                          levels = c('No', 'Yes'),
                          labels = c(1, 2))

# Normalize the numeric columns (so all values are between 0 and 1)
df$avg_glucose_level <- scale(df$avg_glucose_level)
df$bmi <- scale(df$bmi)

# Checking for correlation among independent variables
corrTable <- cor(df[,c("age","hypertension","heart_disease","avg_glucose_level","bmi")])
corrTable
plot_correlation(df,'continuous', cor_args = list("use" = "pairwise.complete.obs"))

# density analysis
plot_density(df)

# Create a histogram
plot_histogram(df)

#convert target varibale to factor
df$stroke <- as.factor(df$stroke)
glimpse(df)

#resampling
# Count the number of samples in each class
class_counts <- table(df$stroke)
table(df$stroke)
# Find the class with the minimum number of samples
minority_class <- which.min(class_counts)

# Calculate the desired number of samples for the majority class
desired_majority_count <- class_counts[minority_class]

# Perform undersampling by randomly selecting samples from the majority class
undersampled_data <- df %>%
  group_by(stroke) %>%
  sample_n(size = desired_majority_count, replace = FALSE) %>%
  ungroup()
ds <- undersampled_data


# Split the dataset into training and testing sets
set.seed(123)
split <- sample.split(ds$stroke, SplitRatio = 0.8)
train_data <- subset(ds, split == TRUE)
test_data <- subset(ds, split == FALSE)
#```````````````````````````````````````````````````````````````````````````````````````````````

# Build a logistic regression model
logistic_model <- glm(stroke ~ ., data = train_data, family = "binomial")

# Make predictions on the test set
predictions <- predict(logistic_model, newdata = test_data, type = "response")

# Evaluate the model
roc_curve <- roc(test_data$stroke, predictions)
auc_value <- auc(roc_curve)

cat("Area Under the Curve (AUC):", auc_value, "\n")

# Convert predictions to factor with levels
predictions <- factor(ifelse(predictions > 0.5, 1, 0), levels = levels(test_data$stroke))

# Convert test_data$stroke to factor with levels
test_data$stroke <- factor(test_data$stroke, levels = levels(predictions))

summary(logistic_model)

# Create confusion matrix
conf_matrix <- confusionMatrix(predictions, test_data$stroke)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate Precision, Recall, and F1 Score
precision <- posPredValue(predictions, reference = test_data$stroke)
recall <- sensitivity(predictions, reference = test_data$stroke)
f1_score <- (2 * precision * recall) / (precision + recall)

cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")

# Calculate AUC for the model
roc_curve_lr <- roc(test_data$stroke, as.numeric(predictions))
auc_value_lr <- auc(roc_curve_optimized_dt)

cat("AUC:", auc_value_lr, "\n")

# Feature selection using LASSO regularization
x <- model.matrix(stroke ~ ., data = train_data)[, -1]
y <- train_data$stroke

library(glmnet)
# Fit LASSO model
lasso_model <- cv.glmnet(x, y, alpha = 1, family = "binomial")

# Set larger margins to avoid "figure margins too large" error
par(mar = c(5, 5, 2, 2))

# Plot LASSO coefficient paths
plot(lasso_model, main = "LASSO Coefficient Paths", xvar = "lambda")

# Extract selected features
selected_features <- coef(lasso_model, s = "lambda.min")[-1, ]
selected_feature_names <- names(selected_features[selected_features != 0])

cat("Selected Features:", selected_feature_names, "\n")

# Assuming selected_feature_names contains the names of the features you want to keep
selected_feature_names <- c("age",  "ever_married2",  "avg_glucose_level")  # Replace ... with actual feature names

# Select only the relevant columns, including the response variable "stroke"
train_data <- train_data %>%
  select(all_of(selected_feature_names))

# Print column names in train_data
print(names(train_data))

# Check if all selected_feature_names exist in the column names
print(selected_feature_names)

# Print summary of LASSO model
lasso_summary <- summary(lasso_model)
print(lasso_summary)


# Build an optimized logistic regression model with selected features
optimized_logistic_model <- glm(stroke ~ ., data = train_data[, c("stroke", selected_feature_names)], family = "binomial")

# Make predictions on the test set with the optimized model
optimized_predictions <- predict(optimized_logistic_model, newdata = test_data[, c("stroke", selected_feature_names)], type = "response")

# Evaluate the optimized model
optimized_roc_curve <- roc(test_data$stroke, optimized_predictions)
optimized_auc_value <- auc(optimized_roc_curve)

cat("Optimized Model AUC:", optimized_auc_value, "\n")

# Convert optimized_predictions to factor with levels
optimized_predictions <- factor(ifelse(optimized_predictions > 0.5, 1, 0), levels = levels(test_data$stroke))

# Create confusion matrix for optimized model
conf_matrix_optimized <- confusionMatrix(optimized_predictions, test_data$stroke)
print("Optimized Model Confusion Matrix:")
print(conf_matrix_optimized)

# Calculate Precision, Recall, and F1 Score for optimized model
precision_optimized <- posPredValue(optimized_predictions, reference = test_data$stroke)
recall_optimized <- sensitivity(optimized_predictions, reference = test_data$stroke)
f1_score_optimized <- (2 * precision_optimized * recall_optimized) / (precision_optimized + recall_optimized)

cat("Optimized Model Precision:", precision_optimized, "\n")
cat("Optimized Model Recall:", recall_optimized, "\n")
cat("Optimized Model F1 Score:", f1_score_optimized, "\n")

#`````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````
# Decision Tree
library(rpart)
library(caret)
library(pROC)

# Build the decision tree model
dt_model <- rpart(stroke ~ ., data = train_data, method = "class")

# Make predictions on the test set
predictions <- predict(dt_model, newdata = test_data, type = "class")

# Evaluate model performance
confusion_matrix_dt <- confusionMatrix(predictions, test_data$stroke)
print("Confusion Matrix:")
print(confusion_matrix_dt)

# Calculate Precision, Recall, and F1 Score
precision_dt <- posPredValue(predictions, reference = test_data$stroke)
recall_dt <- sensitivity(predictions, reference = test_data$stroke)
f1_score_dt <- (2 * precision_dt * recall_dt) / (precision_dt + recall_dt)

cat("Precision:", precision_dt, "\n")
cat("Recall:", recall_dt, "\n")
cat("F1 Score:", f1_score_dt, "\n")

# Model optimization with parameter tuning using caret
ctrl <- trainControl(method = "cv", number = 5)
grid <- expand.grid(cp = seq(0.01, 0.1, by = 0.01))  # Vary the complexity parameter

optimized_model_dt <- train(stroke ~ ., data = train_data, method = "rpart",
                            trControl = ctrl, tuneGrid = grid)

# Print the optimized model details
print("Optimized Model Details:")
print(optimized_model_dt)

# Make predictions on the test set using the optimized model
optimized_predictions_dt <- predict(optimized_model_dt, newdata = test_data)

# Evaluate optimized model performance
optimized_confusion_matrix_dt <- confusionMatrix(optimized_predictions_dt, test_data$stroke)
print("Optimized Model Confusion Matrix:")
print(optimized_confusion_matrix_dt)

# Calculate Precision, Recall, and F1 Score for the optimized model
precision_optimized_dt <- posPredValue(optimized_predictions_dt, reference = test_data$stroke)
recall_optimized_dt <- sensitivity(optimized_predictions_dt, reference = test_data$stroke)
f1_score_optimized_dt <- (2 * precision_optimized_dt * recall_optimized_dt) / (precision_optimized_dt + recall_optimized_dt)

cat("Optimized Model Precision:", precision_optimized_dt, "\n")
cat("Optimized Model Recall:", recall_optimized_dt, "\n")
cat("Optimized Model F1 Score:", f1_score_optimized_dt, "\n")

# Calculate AUC for the optimized model
roc_curve_optimized_dt <- roc(test_data$stroke, as.numeric(optimized_predictions_dt))
auc_value_optimized_dt <- auc(roc_curve_optimized_dt)

cat("Optimized Model AUC:", auc_value_optimized_dt, "\n")
summary(dt_model)
plot(roc_curve_optimized_dt, main = "ROC Curve", col = "blue", lwd = 2)
plot(optimized_model_dt)

#```````````````````````````````````````````````````````````````````````````````````````````````````````
# Random Forest
library(randomForest)
library(caret)
library(pROC)

# Build the Random Forest model
target_variable <- "stroke"
rf_model <- randomForest(formula = as.formula(paste(target_variable, "~ .")), data = train_data)

# Make predictions on the test set
predictions_rf <- predict(rf_model, newdata = test_data)

# Evaluate the initial model
conf_matrix_rf <- confusionMatrix(predictions_rf, test_data$stroke)
print("Initial Model Confusion Matrix:")
print(conf_matrix_rf)

# Calculate Precision, Recall, and F1 Score for the initial model
precision_rf <- posPredValue(predictions_rf, reference = test_data$stroke)
recall_rf <- sensitivity(predictions_rf, reference = test_data$stroke)
f1_score_rf <- (2 * precision_rf * recall_rf) / (precision_rf + recall_rf)

cat("Initial Model Precision:", precision_rf, "\n")
cat("Initial Model Recall:", recall_rf, "\n")
cat("Initial Model F1 Score:", f1_score_rf, "\n")

# Calculate AUC for the initial model
roc_curve_rf <- roc(test_data$stroke, as.numeric(predictions_rf))
auc_value_rf <- auc(roc_curve_rf)

cat("Initial Model AUC:", auc_value_rf, "\n")

# Model optimization with parameter tuning using caret
ctrl <- trainControl(method = "cv", number = 5)
grid <- expand.grid(mtry = c(2, 4, 6))  # Vary the number of variables randomly sampled as candidates at each split

optimized_model_rf <- train(as.formula(paste(target_variable, "~ .")), data = train_data, method = "rf",
                            trControl = ctrl, tuneGrid = grid)

# Print the optimized model details
print("Optimized Model Details:")
print(optimized_model_rf)

# Make predictions on the test set using the optimized model
optimized_predictions_rf <- predict(optimized_model_rf, newdata = test_data)

# Evaluate optimized model performance
optimized_conf_matrix_rf <- confusionMatrix(optimized_predictions_rf, test_data$stroke)
print("Optimized Model Confusion Matrix:")
print(optimized_conf_matrix_rf)

# Calculate Precision, Recall, and F1 Score for the optimized model
precision_optimized_rf <- posPredValue(optimized_predictions_rf, reference = test_data$stroke)
recall_optimized_rf <- sensitivity(optimized_predictions_rf, reference = test_data$stroke)
f1_score_optimized_rf <- (2 * precision_optimized_rf * recall_optimized_rf) / (precision_optimized_rf + recall_optimized_rf)

cat("Optimized Model Precision:", precision_optimized_rf, "\n")
cat("Optimized Model Recall:", recall_optimized_rf, "\n")
cat("Optimized Model F1 Score:", f1_score_optimized_rf, "\n")

# Calculate AUC for the optimized model
roc_curve_optimized_rf <- roc(test_data$stroke, as.numeric(optimized_predictions_rf))
auc_value_optimized_rf <- auc(roc_curve_optimized_rf)

cat("Optimized Model AUC:", auc_value_optimized_rf, "\n")
summary(optimized_model_rf)
plot(optimized_model_rf)


