close all
clear all
clc

norm_mu = 0.003;
norm_sig = 0.00029;

r = norm_sig.*randn(10000,1) + norm_mu;

figure
subplot(2,1,1)
hist(r,100)
xlim([1.5e-3 5.5e-3]); 

lognorm_mu = 0.003;
lognorm_sig = .0000001;
mu = log((lognorm_mu^2)/sqrt(lognorm_sig+lognorm_mu^2));
sigma = sqrt(log(lognorm_sig/(lognorm_mu^2)+1));

X = lognrnd(mu,sigma,1,10000)

subplot(2,1,2)
hist(X,100)
xlim([1.5e-3 5.5e-3]); 