import numpy as np
import math
import matplotlib.pyplot as plt

# Return the (log)likelihood of obs, given the probs
# Binomial Distribution Log PDF
def BN_log_likelihood(obs,probs):

    N = sum(obs);#number of trials  
    k = obs[0] # number of heads

    #### PRACTICE 1 START 
    #
    # complete the computation of the binomail_coeff and prod_probs parameters
    # ln (pdf)      =   Binomial Coeff * product of probabilities
    # ln[f(x|n, p)] =   comb(N,k)    * num_heads*ln(pH) + (N-num_heads) * ln(1-pH)

    binomial_coeff = 0
    prod_probs = 0

    #
    #### PRACTICE 1 END

    log_lik = binomial_coeff + prod_probs

    return log_lik


# Experiments
# 1st:  Coin B, {HTTTHHTHTH}, 5H,5T --> (5,5)

# represent the experiments
head_counts = np.array([5])
tail_counts = 10-head_counts
experiments = zip(head_counts,tail_counts)

print experiments

# maximum number of E-M iterations
maxiter=100

# initialise the pA(heads) and pB(heads) parameters
pA_heads = np.zeros(maxiter); pA_heads[0] = 0.60
pB_heads = np.zeros(maxiter); pB_heads[0] = 0.50

j=0

expectation_A = np.zeros((len(experiments),2), dtype=float) 
expectation_B = np.zeros((len(experiments),2), dtype=float)
for i in range(0,len(experiments)):
    e = experiments[i] # i'th experiment
    
    # loglikelihood of e given coin A:
    ll_A = BN_log_likelihood(e,np.array([pA_heads[j],1-pA_heads[j]])) 
    
    # loglikelihood of e given coin B
    ll_B = BN_log_likelihood(e,np.array([pB_heads[j],1-pB_heads[j]])) 

    print ll_A, ll_B   


    
    #### PRACTICE 2 START 
    #
    # Compute the distributed weight of coin A and coin B based on
    # the likelihoods calculated above
    
    # corresponding weight of A proportional to likelihood of A 
    weightA = 0

    # corresponding weight of B proportional to likelihood of B
    weightB = 0

    expectation_A[i] = 0
    expectation_B[i] = 0


    #
    #### PRACTICE 2 END


#### PRACTICE 3 START 
#
# Maximize pA and pB based on the expected H and T
# frequencies computed above (i.e., expectation_A, expectation_B)

pA_heads[j+1] = 0
pB_heads[j+1] = 0

#
#### PRACTICE 2 END