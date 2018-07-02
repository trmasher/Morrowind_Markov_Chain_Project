# Morrowind Markov Chain Project
This repository contains all of the code, notes, and functions that I am using to perform an investigation on how an interaction by the Player Character, the Nerevarine, with a Non-Player Character (NPC) affects the NPC's disposition towards the Nerevarine. The primary tool that I use to do this is the Markov Chain.

## Major Steps:
*Italicized entries are not yet implemented*
1. Construct the framework of a transition matrix modeling an interaction by the Nerevarine with an NPC that affects the NPC's disposition of the Nerevarine using a Pandas DataFrame object.
2. Use aspects of the geometric series to fill entries of the blank framework matrix to create a valid transition matrix.
3. Use the function 'MarkovState_Calculator()' from the repository "functions.py" to observe evolution of disposition states over chains and study long-term behavior for the two absorbing states.
4. *Perform simulations starting from a certain disposition state over a number of chains.*
5. *Refine the probabilities of the transition matrix to better reflect observed changes in disposition at a specific disposition level for a given interaction.*
