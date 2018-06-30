# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 05:38:55 2018

@author: Travis
"""

import numpy as np
import pandas as pd



#The possible states of admiration a person may have for the Nerevarine.
states = list(range(101))
str_states = [str(item) for item in states]


#To create our transition matrix, we will begin by assigning a 'null probability vector' to each resulting
# probability state j for all 101 starting probability state i.
to_pc_vector = {}
for n in range(101):
    to_pc_vector['{}'.format(str_states[n])] = np.repeat(0,101).tolist()
trans = pd.DataFrame(to_pc_vector, index = str_states)
#To correct the column order in our transition matrix, we simply reorder wrt 'str_states':
trans = trans[str_states]

#To simplify the process of selecting entries of transition matrix, we will define a function that turns
# integer locations into string indices for Python to interpret.
def qk_loc(data,row,col):
    if not set((str(row),str(col))).issubset(set(str_states)):
        return("Either your 'row' or 'col' input could not be coerced to a string within 'data's key.")
    return(data.loc[str(row),str(col)])
    
#To simplify the procedure of calling integer values as strings, we will create a shortcut function:
def s(val):
    return(str(val))


        
#Our first attempt at simulating the entries of our transition matrix will be a bit naive: for initial 
# state i, the probability that a transition to next state j=i will be defined to be 0.5. Further, we will
# define the transition probability from i to k to be zero for any k > i+9; that is, only the next states
# within 9 units of state j=i will be a possible destination. Lastly, we will operate under the reasonable
# assumption that if P_i_j represents the probability of transitioning from initial state i to next state j,
# then P_i_j > P_i_k whenever |i-j|<|i-k|: the further away from state j=i that k is, the smaller its
# transition probability will be.

#To calculate our values, we desire a function whose sum over nonzero values satisfies the above conditions.
# The geometric series is just such a 'function sum' for our purposes. Because we are dealing with discrete
# values, any notion of a mathematical function is out the window; instead, we will make use of a sequence,
# the discrete analogue of a function. Because a series is defined as the summation of a sequence, then what
# we are interested in solving is the sequence of the geometric series.

#We will begin with the general geometric series sum{a*r**(x-1), x in range(1:11)}. Because of the properties
# of geometric series, we know that this sum is equal to [a(1-r**n)]/(1-r). Because we know our sum must be equal
# to 1, we can substitute our sum with 1, and because we know our sequence consists of 10 nonzero values, we
# can substitute n with 10. This leads us to the equality 1=[a(1-r**10)]/(1-r). However, this equality is 
# effectively a 'system' of 1 equation in 2 unknowns, meaning that our system is underdetermined and has no
# 0-dimensional point solution. We can rectify this by establishing another assumption: let P_i_i = 0.5. From
# our sequence, this implies that a*r**(1-1)=0.5 => a=0.5. From this, we can reduce our equation down to a
# single variable r: 2=(1-r**10)/(1-r). Using a computer to approximate solutions, we find two real solutions
# for r: r=1 and r~0.500493; however, if r=1, then (1-r)**-1 is undefined, hence our only reasonable solution
# is r~0.500493. We will proceed as if this approximation is an equality, allowing for trivial error.

#To find the solution set for {xi: i in range(1,11)}, we will define a function seq to represent our geometric
# sequence and then assign a solution vector X to have entries with seq applied over the range(1,11):
seq = lambda x : 0.5*(0.500493)**(x-1)
X = [seq(item) for item in range(1,11)]

#But each entry xl s.t. l != 1 of X does not equal P_i_k, as there are TWO such entries of our transition
# matrix that lie within distance k of next state j=i. We fix this by defining vector P where element pk of P
# s.t. k != 1 is equal to half ot what element xk of vector X equals; that is, P=[x1]v[0.5*xk: k in 
# range(2,11)]:
P = [X[0]]
for k in range(1,10):
    P.append(0.5*X[k])

    
#With the development of our probability distribution formula for a given initial state i is developed, we now
# will now find a way to, for every initial state i, distribute probability vector P to the appropriate next
# states j across the columns of our transition matrix.
def dist_P(trans,P):
    for i in range(0,101):
        trans.loc[s(i),s(i)] = P[0]
        for k in range(1,10):
            # We perform a check to see if entry P_i_j is a valid entry in our transition matrix whenever
            # j=i-k or j=i+k
            # If both entries within distance k of i are valid entries...
            if all((i - k >= 0, i + k <= 100)):
                trans.loc[s(i),s(i-k)] = P[k]
                trans.loc[s(i),s(i+k)] = P[k]
            # If k is s.t. j < 0 ...
            elif i - k < 0:
                trans.loc[s(i),s(i+k)] = X[k]
            # If k is s.t. j > 100 ...
            elif i + k > 100:
                trans.loc[s(i),s(i-k)] = X[k]
            # Because max(k) = 9 is not greater than or equal to half of the rank of our transition matrix,
            # it should never happen that both of the above indices are out of bounds...
            else:
                return("An error has occured.")
                
dist_P(trans,P)
#To make sure function dist_P() correctly applied probability values, we will calculate all unique values of
# row sums and make sure that they differ from 1 by less than 5 decimal digits:
rows = [trans.loc[s(j),] for j in range(101)]
row_sums = [sum(item) for item in rows]
unq_row_sums = list(set(row_sums))

if all([round(entry,5) == 1 for entry in unq_row_sums]):
    print("All sums are within 1e-5 of 1.")

            