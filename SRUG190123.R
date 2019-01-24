###
# If you don't have all the required packages installed, you can try the following
###
unavailable <- setdiff(c('DALEX', 'dplyr', 'iml', 'pdp', 'randomForest',
                         'randomForestExplainer', 'xgboost'), 
                          rownames(installed.packages()))
if(length(unavailable)>0){install.packages(unavailable)}

if (!require('xgboostExplainer')){
  if(!require('devtools')){
    install.packages('devtools')
  }
  library('devtools') 
  install_github('AppliedDataSciencePartners/xgboostExplainer')
}


###
# Load packages
###

library('DALEX')
library('dplyr')
library('iml')
library('pdp')
library('randomForest')
library('randomForestExplainer')
library('xgboost')
library('xgboostExplainer')


data('Boston', package='MASS')
#CRIM - per capita crime rate by town
#ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
#INDUS - proportion of non-retail business acres per town.
#CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
#NOX - nitric oxides concentration (parts per 10 million)
#RM - average number of rooms per dwelling
#AGE - proportion of owner-occupied units built prior to 1940
#DIS - weighted distances to five Boston employment centres
#RAD - index of accessibility to radial highways
#TAX - full-value property-tax rate per $10,000
#PTRATIO - pupil-teacher ratio by town
#B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#LSTAT - % lower status of the population
#MEDV - Median value of owner-occupied homes in $1000's
set.seed(42)

###
# Feature importance.
###

# Out-of-the-box RF variable importance.
# cf. https://explained.ai/rf-importance/index.html
rf <- randomForest(medv ~ ., data = Boston, localImp = TRUE, ntree = 100)
varImpPlot(rf)

# RandomForestExplainer
# https://cran.rstudio.com/web/packages/randomForestExplainer/vignettes/randomForestExplainer.html
min_depth_frame <- min_depth_distribution(rf)
plot_min_depth_distribution(min_depth_frame)

###
# Feature influence on target
###

# pdp (partial dependence plot)
# https://christophm.github.io/interpretable-ml-book/pdp.html
partialPlot(rf, pred.data = Boston, x.var='lstat')
# or ...
pdp::partial(rf, pred.var = 'lstat', plot=T, plot.engine = 'ggplot2')
# ICE
# https://christophm.github.io/interpretable-ml-book/ice.html
pdp::partial(rf, pred.var = 'lstat', plot=T, ice = T, plot.engine = 'ggplot2')


# ALE (accumulated local effect) plot
# https://christophm.github.io/interpretable-ml-book/ale.html
# with iml
# https://cran.r-project.org/web/packages/iml/vignettes/intro.html
predictor = Predictor$new(rf, data = Boston %>% select(-medv), y = Boston$medv)
ale = FeatureEffect$new(predictor, feature='lstat')
plot(ale)

# with DALEX
# https://pbiecek.github.io/DALEX_docs/index.html#introduction
explainer_rf <- explain(rf, data = Boston %>% select(-medv), y = Boston$medv)
rf_pdp  <- single_variable(explainer_rf, variable =  "lstat", type = "pdp")
plot(rf_pdp)
rf_ale  <- single_variable(explainer_rf, variable =  "lstat", type = "ale")
plot(rf_ale)

# Categorical variables.
View(apartments) # Synthetic dataset from DALEX with apartments in Warsaw.
apartments_rf_model <- randomForest(m2.price ~ construction.year + surface + floor + 
                                      no.rooms + district, data = apartments)
explainer_rf <- explain(apartments_rf_model, 
                        data = apartmentsTest[,2:6], y = apartmentsTest$m2.price)
cat_rf  <- single_variable(explainer_rf, variable =  "district", type = "pdp")
plot(cat_rf)

# Feature interactions
interact = Interaction$new(predictor)
# Overall interaction strength
plot(interact)
interact = Interaction$new(predictor, feature='lstat')
plot(interact)


###
# Prediction explanation
###

# model-dependent: xgboostExplainer
# https://medium.com/applied-data-science/new-r-package-the-xgboost-explainer-51dd7d1aa211
X <- Boston %>% select(-medv) %>% as.matrix()
y <- Boston %>% select(medv) %>% as.matrix()

random_exs <- X[sample(nrow(X), 5),]

xgb.data <- xgb.DMatrix(X, label = y, missing = NA)
xgb.model <- xgboost(data=xgb.data, nrounds = 100)
# yet another importance plot is possible ...
# xgb.plot.importance(importance_matrix = xgb.importance(model=xgb.model), top_n = 10)

explainer = buildExplainer(xgb.model, xgb.data, type="linear", base_score = 0.5, trees_idx = NULL)
to_pred <- xgb.DMatrix(as.matrix(random_exs))
pred.breakdown = explainPredictions(xgb.model, explainer, to_pred)
showWaterfall(xgb.model, explainer, to_pred, as.matrix(random_exs), 1, type='linear')
showWaterfall(xgb.model, explainer, to_pred, as.matrix(random_exs), 2, type='linear')

# model-agnostic explanations

# lime in iml
# LIME paper https://arxiv.org/abs/1602.04938
locmod = LocalModel$new(predictor, x.interest = data.frame(random_exs)[1,])
plot(locmod)

# game theory
# https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d
shapley = Shapley$new(predictor, x.interest = data.frame(random_exs)[1,])
shapley$plot()

###
# Extra DALEX stuff
###

# Linear regression
apartments_lm_model <- lm(m2.price ~ construction.year + surface + floor + 
                            no.rooms + district, data = apartments)
predicted_mi2_lm <- predict(apartments_lm_model, apartmentsTest)
sqrt(mean((predicted_mi2_lm - apartmentsTest$m2.price)^2))

explainer_lm <- explain(apartments_lm_model, data = apartmentsTest[,2:6], y = apartmentsTest$m2.price)

# RF
apartments_rf_model <- randomForest(m2.price ~ construction.year + surface + floor + 
                                      no.rooms + district, data = apartments)
apartments_rf_model

predicted_mi2_rf <- predict(apartments_rf_model, apartmentsTest)
sqrt(mean((predicted_mi2_rf - apartmentsTest$m2.price)^2))

explainer_rf <- explain(apartments_rf_model, 
                        data = apartmentsTest[,2:6], y = apartmentsTest$m2.price)

# model_performance() function of DALEX
mp_lm <- model_performance(explainer_lm) # weird bimodal distribution of residuals. interactions?
mp_rf <- model_performance(explainer_rf)
plot(mp_lm, mp_rf)

# variable importance
vi_rf <- variable_importance(explainer_rf, loss_function = loss_root_mean_square)
vi_lm <- variable_importance(explainer_lm, loss_function = loss_root_mean_square)
plot(vi_rf, vi_lm)

# single variable (pdp)
sv_rf  <- single_variable(explainer_rf, variable =  "construction.year", type = "pdp")
sv_lm  <- single_variable(explainer_lm, variable =  "construction.year", type = "pdp")
plot(sv_rf, sv_lm) # perhaps the residuals come from the failure of lm here


# prediction explanations
# already have mp_rf from above
ggplot(mp_rf, aes(observed, diff)) + geom_point() + 
  xlab("Observed") + ylab("Predicted - Observed") + 
  ggtitle("Diagnostic plot for the random forest model") + theme_mi2()

which.min(mp_rf$diff)
## 1161
new_apartment <- apartmentsTest[which.min(mp_rf$diff), ]
new_apartment

new_apartment_rf <- single_prediction(explainer_rf, observation = new_apartment)
breakDown:::print.broken(new_apartment_rf)
plot(new_apartment_rf)

new_apartment_lm <- single_prediction(explainer_lm, observation = new_apartment)
plot(new_apartment_lm, new_apartment_rf)
