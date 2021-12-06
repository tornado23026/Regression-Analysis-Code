insurance <- read_csv("insurance.csv")


rmse = function(a, b) {
  return( sqrt( mean( (a - b)^2 ) ) )
}



data = insurance


sex_binary = as.integer(insurance$sex == 'female') 
smoke_binary = as.integer(insurance$smoker == 'yes') 

data$sex = sex_binary
data$smoker = smoke_binary


data$northeast = as.integer(insurance$region == 'northeast')
data$northwest = as.integer(insurance$region == 'northwest')
data$southeast = as.integer(insurance$region == 'southeast')

data = subset(data, select = -region)

hist(data$charges, breaks=20)


cor = cor(data)

par(mfrow=c(1, 1))
corrplot(cor, method='color')


ggplot(data, aes(x=charges, y=stat(density))) + 
  geom_histogram(data = subset(data, smoker == 1), color='black', fill="gold", bins = 25, alpha=0.5) +
  geom_density(data = subset(data, smoker == 1), col='gold4', size=1) +
  geom_histogram(data = subset(data, smoker == 0), color='black', fill="darkgreen", bins = 25, alpha=0.5) + 
  geom_density(data = subset(data, smoker == 0), col='darkgreen', size=1) 

set.seed(69420)
index = sample(nrow(data), as.integer(0.8 * nrow(data)))

train = data[index,]
test = data[-index,]

baseline = lm(log(charges) ~ ., data = train)
summary(baseline)

par(mfrow=c(2, 2))
plot(baseline)

resid_panel(baseline)



smoker = train$smoker == 1

train_non_smoker = train[!smoker,]
train_smoker = train[smoker,]

mod1 = lm(log(charges) ~ . - smoker, data = train_non_smoker)
mod2 = lm(log(charges) ~ . - smoker, data = train_smoker)

summary(mod1)
summary(mod2)

resid_panel(mod1)

resid_panel(mod2)


rmse(predict(baseline, test), log(test$charges))


test.smoke = test$smoker == 1

test_non_smoker = test[!test.smoke,]
test_smoker = test[test.smoke,]


sqrt((sum((predict(mod1, test_non_smoker) - log(test_non_smoker$charges))^2) + 
  sum((predict(mod2, test_smoker) - log(test_smoker$charges))^2)) / nrow(test))



xtrain = subset(train, select = -c(charges))
ytrain = train[,'charges']

xtest = subset(test, select = -c(charges))
ytest = test[,'charges']



xgb.model = xgboost(data = as.matrix(xtrain), 
                     label = as.matrix(log(ytrain)), max_depth = 6,
                     eta = 0.05, nrounds = 5000,
                     subsample = 0.5,
                     verbose = FALSE)
rmse(predict(xgb.model, as.matrix(xtest)), log(test$charges))
     
par(mfrow = c(1, 1))

tree = rpart(log(charges) ~ ., data = train)
plotcp(tree)

plot(tree, uniform=TRUE)
text(tree, use.n=TRUE, all=TRUE, cex=.8)

set.seed(69420)
rmse(predict(tree, test), log(test$charges))



smoker = train$smoker < 1

non_smoker = train[smoker,]
smoker = train[!smoker,]

age = non_smoker$age < 33.5

non_smoker1 = non_smoker[age,]
non_smoker2 = non_smoker[!age,]


bmi = smoker$bmi < 30.1

smoker1 = smoker[bmi,]
smoker2 = smoker[!bmi,]

m1 = lm(log(charges) ~ . - smoker, data = non_smoker1)
m2 = lm(log(charges) ~ . - smoker, data = non_smoker2)
m3 = lm(log(charges) ~ . - smoker, data = smoker1) 
m4 = lm(log(charges) ~ . - smoker, data = smoker2)


test.smoke = test$smoker == 1

test_non_smoker = test[!test.smoke,]
test_smoker = test[test.smoke,]

test.age = test_non_smoker$age < 33.5

test_non_smoker1 = test_non_smoker[test.age,]
test_non_smoker2 = test_non_smoker[!test.age,]

test.bmi = test_smoker$bmi < 30.1

test_smoker1 = test_smoker[test.bmi,]
test_smoker2 = test_smoker[!test.bmi,]

p1 = predict(m1, test_non_smoker1)
p2 = predict(m2, test_non_smoker2)
p3 = predict(m3, test_smoker1)
p4 = predict(m4, test_smoker2)


sqrt((sum((p1 - log(test_non_smoker1$charges))^2) +
        sum((p2 - log(test_non_smoker2$charges))^2) +
        sum((p3 - log(test_smoker1$charges))^2) + 
        sum((p4 - log(test_smoker2$charges))^2)) / nrow(test))

par(mfrow=c(2, 2))
plot(m1)
plot(m2)
plot(m3)
plot(m4)

summary(m1)
summary(m2)
summary(m3)
summary(m4)





library(MASS)

mr1 = rlm(log(charges) ~ . - smoker, data = non_smoker1)
mr2 = rlm(log(charges) ~ . - smoker, data = non_smoker2)
mr3 = rlm(log(charges) ~ . - smoker, data = smoker1) 
mr4 = rlm(log(charges) ~ . - smoker, data = smoker2)

pr1 = predict(m1, test_non_smoker1)
pr2 = predict(m2, test_non_smoker2)
pr3 = predict(m3, test_smoker1)
pr4 = predict(m4, test_smoker2)


sqrt((sum((pr1 - log(test_non_smoker1$charges))^2) +
        sum((pr2 - log(test_non_smoker2$charges))^2) +
        sum((pr3 - log(test_smoker1$charges))^2) + 
        sum((pr4 - log(test_smoker2$charges))^2)) / nrow(test))


w = Winsorize(train$charges, probs=c(0, 0.95))
wtrain = train
wtrain$charges = w

baseline2 = lm(log(charges) ~ ., data = wtrain)
plot(baseline2)




grid = expand.grid(max_depth = seq(3, 6, 1), eta = seq(.2, .35, .01))

xgb_train_rmse = rep(0, nrow(grid))
xgb_test_rmse = rep(0, nrow(grid))


for (j in 1:nrow(grid)) {
  set.seed(69420)
  m_xgb_untuned <- xgb.cv(
    data = as.matrix(subset(train, select = -c(charges))),
    label = as.matrix(log(train[,'charges'])),
    nrounds = 1000,
    objective = "reg:squarederror",
    early_stopping_rounds = 3,
    nfold = 5,
    max_depth = grid$max_depth[j],
    eta = grid$eta[j]
  )
  
  xgb_train_rmse[j] <- m_xgb_untuned$evaluation_log$train_rmse_mean[m_xgb_untuned$best_iteration]
  xgb_test_rmse[j] <- m_xgb_untuned$evaluation_log$test_rmse_mean[m_xgb_untuned$best_iteration]
  
  cat(j, "\n")
}    

which.min(xgb_test_rmse)


xgb_mod = xgboost(
  data = as.matrix(subset(train, select = -c(charges))),
  label = as.matrix(log(train[,'charges'])),
  nrounds = 1000,
  objective = "reg:squarederror",
  early_stopping_rounds = 3,
  max_depth = 3,
  eta = .27
)   

pred = predict(xgb_mod, as.matrix(subset(test, select = -c(charges))))

rmse(pred, log(test$charges))

par(mfrow=c(1, 1))
plot(log(test$charges) - pred, ylab = 'residuals')

xgb.plot.tree(model = xgb_mod, trees = 0:2)


importance_matrix = xgb.importance(model = xgb_mod)
xgb.plot.importance(importance_matrix, xlab = "Feature Importance")
