
library(pcalg)

setwd("/Users/louksmalbil/Desktop/CDVAE/R/ArchivedData/Hidden_Var_Data")
data <- read.csv(file = 'combined_hidden_nn_3.csv', header = FALSE)
data <- data[2:18]
head(data)

n <- nrow(data)
V <- colnames(data)

suffStat <- list(C = cor(data), n = n)

start_time <- Sys.time()

fci.fit <- fci(suffStat,
               indepTest = gaussCItest, ## indep.test: partial correlations
               alpha=0.01, labels = V, verbose = TRUE)

end_time <- Sys.time()

setwd("/Users/louksmalbil/Desktop/CDVAE/R/")
write.csv(amat, 'predicted.csv')

end_time - start_time
