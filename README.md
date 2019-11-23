# raote-planner

This is a project that will tackle an NP-hard problem.

## Problem statement:

Professor Rao and his army of TAs are working in Soda Hall late at night, writing the final exam for CS 170. Rao offers to drive and drop TAs off closer to their homes so that they can all get back safe despite the late hours. However, the roads are long, and Rao would also like to get back to Soda as soon as he can. Can you plan transportation so that everyone can get home as efficiently as possible?

Formally, you are given an undirected graph *G = (L,E)* where each vertex in *L* is a location. You are also given a starting location *s*, and a list *H* of unique locations that correspond to homes. The weight of each edge *(u, v)* is the length of the road between locations *u* and *v*, and each home in *H* denotes a location that is inhabited by a TA. Traveling along a road takes energy, and the amount of energy expended is proportional to the length of the road. For every unit of distance traveled, the driver of the car expends *2/3* units of energy, and a walking TA expends *1* unit of energy. The car must start and end at *s*, and every TA must return to their home in *H*.

You must return a list of vertices *v_i* that is the tour taken by the car (cycle with repetitions allowed), as well as a list of drop-off locations at which the TAs get off. You may only drop TAs off at vertices visited by the car, and multiple TAs can be dropped off at the same location. We’d like you to produce a route and sequence of drop-offs that minimizes total energy expenditure, which is the sum of Rao’s energy spent driving and the total energy that all of the TAs spend walking. Note TAs do not expend any energy while sitting in the car. You may assume that the TAs will take the shortest path home from whichever location they are dropped off at.
