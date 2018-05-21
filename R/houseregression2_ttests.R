y=log(housing$price)
xt=housereg_test$coefficients%*%(c(1,log(5000),2,2,1))+1

lnlotsize=log(housing$lotsize)-log(5000)
bedrms=housing$bedrooms-2
bathrms=housing$bathrms-2
airco=housing$airco-1
const=rep(1,length(airco))

x=cbind(const,lnlotsize,bedrms,bathrms,airco)

XprimeX=t(x)%*%x
XprimeXinv=solve(XprimeX)
XprimeY=t(x)%*%y
betahat=XprimeXinv%*%XprimeY

housereg=lm(y~lnlotsize+bedrms+bathrms+airco)
summary(housereg)

epsilonhat=y-x%*%betahat
sigma2hat=(t(epsilonhat)%*%epsilonhat)/(length(y)-ncol(x))
varbetahat=sigma2hat[1,1]*XprimeXinv
sebetahat=sqrt(diag(varbetahat))



zbetadiff=(betahat[4]-betahat[3])/sqrt(varbetahat[4,4]+varbetahat[3,3]-2*varbetahat[3,4])


bedbath=bedrms+bathrms
housereg_test=lm(y~lnlotsize+bedbath+bathrms+airco)
summary(housereg_test)


housereg_testr=lm(y~airco)
anova(housereg_test,housereg_testr) #analysis of var


housereg_test2=lm(y~(lnlotsize-log(5000))+(bedbath-2)+(bathrms-2)+(airco-1))
c


