library("lpSolveAPI")

optimize <- function(H, risk_measure="AVaR", alpha=.01, return=0) {
    mu <- vector("numeric")
    k <- nrow(H)
    n <-ncol(H)
    for (col_num in 1:n) {
        col <- H[ , col_num]
        mu[col_num] <- mean(col)
    }

    lp <- make.lp(0, n+1+k)

# objective function
    obj <- vector("numeric")
    for (j in 1:n) {
        obj[j] <- 0
    }

    obj[n+1] <- 1
    for (j in (n+2):(n+1+k)) {
    obj[j] <- 1/(k*(1-alpha))
    }

    set.objfn(lp, obj)

# constraints
    for (i in 1:k) {
        cnstr <- vector("numeric")
        for (j in 1:n) {
            cnstr[j] <- -H[i, j]
        }

        cnstr[n+1] <- -1
        if (2 <= i) {
            for (j in (n+2):(n+i)) {
                cnstr[j] <- 0
            }
        }

        cnstr[n+1+i] <- -1
        if (i<k) {
            for (j in (n+2+i):(n+1+k)) {
                cnstr[j] <- 0
            }
        }

        add.constraint(lp, cnstr, "<=", 0)
    }


    for (i in 1:n) {
        cnstr <- vector("numeric")
        for (j in 1:(n+1+k)) {
            if (i==j) {
                cnstr[j] <- 1
            }
            else {
                cnstr[j] <- 0
            }
        }

        add.constraint(lp, cnstr, ">=", 0)
    }

    for (i in (n+2):(n+1+k)) {
        cnstr <- vector("numeric")
        for (j in 1:(n+1+k)) {
            if (i==j) {
                cnstr[j] <- 1
            }
            else {
                cnstr[j] <- 0
            }
        }

        add.constraint(lp, cnstr, ">=", 0)
    }

    cnstr <- vector("numeric")
    for (j in 1:n) {
        cnstr[j] <- mu[j]
    }
    for (j in (n+1):(n+1+k)) {
        cnstr[j] <- 0
    }

    add.constraint(lp, cnstr, ">=", return)

    for (j in 1:n) {
        cnstr[j] <- 1
    }

    add.constraint(lp, cnstr, "=", 1)

    return(lp)
}
