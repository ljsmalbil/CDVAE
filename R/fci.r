


library(pcalg)

setwd("/Users/louksmalbil/Desktop/CDVAE/R/ArchivedData")
data <- read.csv(file = 'combined_nn_5.csv', header = FALSE)
data <- data[2:21]
head(data)

n <- nrow(data)
V <- colnames(data)

suffStat <- list(C = cor(data), n = n)

start_time <- Sys.time()

fci.fit <- fci(suffStat,
               indepTest = gaussCItest, ## indep.test: partial correlations
               alpha=0.01, labels = V, verbose = TRUE)

end_time <- Sys.time()

as(fci.cpdag, "amat")
amat <- wgtMatrix(fci.fit, transpose = TRUE)
amat <- replace(amat, amat == 1, 1)
amat <- replace(amat, amat == 2, 0)
amat <- replace(amat, amat == 3, 1)
amat <- t(amat)

setwd("/Users/louksmalbil/Desktop/CDVAE/R/")
write.csv(amat, 'predicted.csv')

end_time - start_time
