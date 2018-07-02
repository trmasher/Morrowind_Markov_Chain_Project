# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 05:38:55 2018

@author: Travis
"""

import numpy as np
import pandas as pd
import os
os.chdir('D:\\`IntPy')
from functions import MarkovState_Calculator

## Before we begin, we will need to establish some basic terminology to describe the elements of our data here
#   and their correspondence to elements and objects in The Elder Scrolls III: Morrowind. The term 'Player
#   Character', or 'PC', refers to the protagonist of the game that is in control of the player; in this game,
#   because the Player Character is often referred to as 'the Nerevarine', we use that term instead. The term 
#   'Non-Player Character', or 'NPC', refers to all of the other human or human-like entities other than the
#   Nerevarine in the game. The term 'Disposition' refers to the percentage value assigned to a given
#   NPC that represents the NPC's current degree of affinity towards the Nerevarine. 
# By interacting in certain ways with an NPC, the Nerevarine can alter the NPCs disposition towards them; 
#   furthermore, from observation, the state of disposition that the NPC changes to after an interaction is 
#   dependant upon the state of disposition that they were in prior to the interaction. Because the amount of
#   of time that elapses between adjacent interactions do not in any way affect the change in disposition the 
#   subsequent interaction precipitates, the process satisfies the property of 'memorylessness' and hence can
#   be modeled by a Markov Chain.
# In the program below, I will attempt to model the transition matrix that corresponds to the Markov Chain
#   which represents disposition changes of NPCs towards the Nerevarine. I will then investigate the 
#   properties and long-term behavior of the model.


# The possible states of admiration a person may have for the Nerevarine:
states = list(range(101))
str_states = [str(item) for item in states]


# To create our transition matrix, we will begin by assigning a 'null probability vector' to each resulting
#   probability state j for all 101 starting probability state i.
to_pc_vector = {}
for n in range(101):
    to_pc_vector['{}'.format(str_states[n])] = np.repeat(0,101).tolist()
trans = pd.DataFrame(to_pc_vector, index = str_states)
#To correct the column order in our transition matrix, we simply reorder wrt 'str_states':
trans = trans[str_states]

#To simplify the process of selecting entries of transition matrix, we will define a function that turns
#   integer locations into string indices for Python to interpret.
def qk_loc(data,row,col):
    if not set((str(row),str(col))).issubset(set(str_states)):
        return("Either your 'row' or 'col' input could not be coerced to a string within 'data's key.")
    return(data.loc[str(row),str(col)])
    
#To simplify the procedure of calling integer values as strings, we will create a shortcut function:
def s(val):
    return(str(val))


# Our first attempt at simulating the entries of our transition matrix will be a bit naive: for initial 
#   state i, the probability that a transition to next state j=i will be defined to be 0.5. Further, we will
#   define the transition probability from i to k to be zero for any k > i+9; that is, only the next states
#   within 9 units of state j=i will be a possible destination. Lastly, we will operate under the reasonable
#   assumption that if P_i_j represents the probability of transitioning from initial state i to next state j,
#   then P_i_j > P_i_k whenever |i-j|<|i-k|: the further away from state j=i that k is, the smaller its
#   transition probability will be.

# To calculate our values, we desire a function whose sum over nonzero values satisfies the above conditions.
#   The geometric series is just such a 'function sum' for our purposes. Because we are dealing with discrete
#   values, any notion of a mathematical function is out the window; instead, we will make use of a sequence,
#   the discrete analogue of a function. Because a series is defined as the summation of a sequence, then what
#   we are interested in solving is the sequence of the geometric series.

# We will begin with the general geometric series sum{a*r**(x-1), x in range(1:11)}. Because of the properties
#   of geometric series, we know that this sum is equal to [a(1-r**n)]/(1-r). Because we know our sum must be
#   equal to 1, we can substitute our sum with 1, and because we know our sequence consists of 10 nonzero 
#   values, we can substitute n with 10. This leads us to the equality 1=[a(1-r**10)]/(1-r). However, this 
#   equality is effectively a 'system' of 1 equation in 2 unknowns, meaning that our system is underdetermined
#   and has no 0-dimensional point solution. We can rectify this by establishing another assumption: let 
#   P_i_i = 0.5. From our sequence, this implies that a*r**(1-1)=0.5 => a=0.5. From this, we can reduce our 
#   equation down to a single variable r: 2=(1-r**10)/(1-r). Using a computer to approximate solutions, we 
#   find two real solutions for r: r=1 and r~0.500493; however, if r=1, then (1-r)**-1 is undefined, hence our
#   only reasonable solution is r~0.500493. We will proceed as if this approximation is an equality, allowing
#   for trivial error.

# To find the solution set for {xi: i in range(1,11)}, we will define a function seq to represent our geometric
#   sequence and then assign a solution vector X to have entries with seq applied over the range(1,11):
seq = lambda x : 0.5*(0.500493)**(x-1)
X = [seq(item) for item in range(1,11)]

# But each entry xl s.t. l != 1 of X does not equal P_i_k, as there are TWO such entries of our transition
#   matrix that lie within distance k of next state j=i. We fix this by defining vector P where element pk of
#   P s.t. k != 1 is equal to half ot what element xk of vector X equals; that is, P=[x1]v[0.5*xk: k in 
#   range(2,11)]:
P = [X[0]]
for k in range(1,10):
    P.append(0.5*X[k])

    
# With the development of our probability distribution formula for a given initial state i is developed, we
#   now will now find a way to, for every initial state i, distribute probability vector P to the appropriate
#   next states j across the columns of our transition matrix.
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
# To make sure function dist_P() correctly applied probability values, we will calculate all unique values of
#   row sums and make sure that they differ from 1 by less than 5 decimal digits:
rows = [trans.loc[s(j),] for j in range(101)]
row_sums = [sum(item) for item in rows]
unq_row_sums = list(set(row_sums))

if all([round(entry,5) == 1 for entry in unq_row_sums]):
    print("All sums are within 1e-5 of 1.")
# The above does indeed print, letting us know that all of our row sums are within 0.001% of 1: an acceptable
#   margin of error for our puposes.
    
    
# Before we begin performing simulations using our transition matrix, we would like to establish one additional
#   restriction to our matrix: once a value has reached the state representing 0% or the state representing
#   100%, that state will no longer transition to any other state. This property is referred to as the state
#   being an 'absorbing chain' in the transition matrix and is represented by altering the probability of
#   transitioning from the absorbing state i to any other state j to be 0 for any j!=i and 1 iff j=i.
for item in str_states:
    for i in [0,100]:
        if trans.loc[s(i),item] != 0:
            trans.loc[s(i),item] = 0
        trans.loc[s(i),s(i)] = 1


# Now that we have created our naive transition matrix, we can begin simulating Markov Chain applications. We
#   will make use of my function 'MarkovState_Calculator()' to perform iterations for us. Let's start with the
#   assumption that the Nerevarine and a respective NPC start with a disposition of 25%. We will create an
#   initial state vector from this and then perform 10 Markov Chains:
def single_state_init(val):
    #To simplify setting up initial state vectors, we will use this function to perform it for us.
    try:
        int(val)
    except ValueError:
        return("Your input is not valid.")
    if int(val) < 0 or int(val) > 100:
        return("Your input is not valid.")
    init_state = [0]*101
    init_state[val] = 1
    return init_state

start = single_state_init(25)
#print(MarkovState_Calculator(trans, start, 10))
# This output is hard to read. We will store the result, then create a function to match create a dictionary
#   between each percentage and the Markov Chain's percentage for each iteration:

M = MarkovState_Calculator(trans, start, 10, 4)
def chains_to_dict(M,index):
    #We check conditions on M
    if type(M) != list:
        return("Input 'M' must be a list.")
    nested_list_check = True
    if type(M[0]) != list:
        if type(M[0]) not in [int,float]:
            return("Entries of list 'M' must be either numerical values or lists themselves.")
        nested_list_check = False
    dic = {}
    if nested_list_check == True:
        if len(M[0]) != len(index):
            return("Entries of input 'M' and input 'index' must have the same length.")
        for val in index:
            try:
                int(val)
            except ValueError:
                return("Your index has entries that cannot be converted to integers.")
        for val in index:
            val_chain = []
            for i in range(0,len(M)):
                val_chain.append(M[i][int(val)])
            dic[val] = val_chain
        return(dic)
    else:
        for val in index:
            try:
                int(val)
            except ValueError:
                return("Your index has entries that cannot be converted to integers.")
        for val in index:
            dic[val] = M[int(val)]
        return(dic)

dic = chains_to_dict(M,str_states)
# Let's output our new dictionary in a readable format. As always, we use a function to generalize this
#   process:
def pretty_dict_print(dic):
    if type(dic) != dict:
        return("Input is not a dictionary.")
    print('_'*70)
    for item in dic:
        #We choose to not print the entries that are all zero.
        try:
            zeroes = [0]*len(dic[item])
        except TypeError:
            zeroes = 0
        if dic[item] != zeroes:
            print('\n',"[['{}%']]: ".format(item),dic[item],'_'*70)
            
pretty_dict_print(dic)


# Let's investigate what happens to our state vectors of disposition liklihoods for large chains:
M1 = MarkovState_Calculator(trans, start, 10000, 5)
dic1 = chains_to_dict(M1,str_states)
pretty_dict_print(dic1)
# As expected, our chains have been absorbed into each steady state! The liklihood of being at disposition 0%
#   with an NPC after starting at a disposition of 25% using our naive approach is 70.52%, and the liklihood
#   of being at disposition 100% with the NPC is 29.46%. Although our percentages are slightly under 100%, we
#   are at least confident with out percentages within a 0.024% margin of error.

# Let's try starting at exactly 50% disposition with an NPC and observe what happens:
start2 = single_state_init(50)
M2 = MarkovState_Calculator(trans, start2, 10000, 5)
dic2 = chains_to_dict(M2,str_states)
pretty_dict_print(dic2)

# We see that, when starting at the exact midpoint of the possible states of disposition availab, we end up
#   with the liklihood of being in either absorbing state is exactly equal (and within a 0.03% margin of
#   error).

# What about situations where the initial disposition state could be a certain value amongst many possibilities?
#   Suppose that a given town consists of NPCs that are all equally like to begin at disposition levels of
#   35%, 40%, 46%, and 53%. Given an encounter by the Nerevarine with a random NPC from this town, what is the
#   long-term disposition probability vector?
init_town_states = [35,40,46,53]
def multi_state_init(group,prob_group=-1):
    #We perform a slight change to the function 'single_state_init' to handle multiple states. If the
    #probability of each state is equally likely, a second list input 'prob_group' need not be passed.
    if type(group) == int:
        if int(group) < 0 or int(group) > 100:
            return("Your input 'group' is not valid.")
        init_state = [0]*101
        init_state[group] = 1
        return(init_state)
    elif type(group) == list:
        #We need to check some conditions on 'prob_group'
        if type(prob_group) != list:
            if prob_group != -1:
                return("Your input 'prob_group' must be a list of numerical values.")
        elif len(group) != len(prob_group):
            return("Your input 'group' must be the same length as your input 'prob_group'.")
        init_state = [0]*101
        check = True
        if prob_group == -1:
            check = False
            prob_val = 1/len(group)
        for item in range(len(group)):
            if check == True:
                if type(prob_group[item]) not in [int,float]:
                    return("The entries of 'prob_group' must be numerical.")
                prob_val = prob_group[item]
            if int(group[item]) < 0 or int(group[item]) > 100:
                return("Your input 'group' is not valid.")
            init_state[group[item]] = prob_val
        if check == True:
            if sum(prob_group) != 1:
                return("The entries of 'pro_group' must sum to 1.")
        return(init_state)
    else:
        return("Your input 'group' is not valid.")
            
start3 = multi_state_init(init_town_states)
M3 = MarkovState_Calculator(trans, start3, 10000, 5)
dic3 = chains_to_dict(M3,str_states)
pretty_dict_print(dic3)
# Apart from requiring the setup of a new function to handle multiple starting probabilities, this disposition
#   Markov Chain was no different than those before it. Similarly, starting at any number of states with a
#   given probability for each state, we can proceed in the same manner.
init_town_states1 = [7,26,37,57,98]
init_town_prob1 = [0.09,0.15,0.44,0.02,0.3]
start4 = multi_state_init(init_town_states1,init_town_prob1)
M4 = MarkovState_Calculator(trans, start4, 10000, 5)
dic4 = chains_to_dict(M4,str_states)
pretty_dict_print(dic4)


# We will now create a function which simulates a sample disposition transition vector using our above matrix:
#def sample_disposition_states(start,trans):
#    if start != list:
#        start_vector = single_state_init(start)
#        if type(start_vector) != list:
#            return("Input 'start' must be either a starting state or a starting state vector for a single"
#                   " state.")
#    else:
#        start_vector = start
#    if start.count(1) != 1 or start.count(0) != len(trans):
#        return("Input 'start' must be either a starting state or a starting state vector for a single state.")
#    if start not in trans.index:
#        if type(start) == int:
#            try:
#                int(start)
#            except
#    master_track = [start]

## Now, although we are satisfied in the fact that our naive transition matrix has provided us with logical
#   and interesting outputs, there is still more to be done. 