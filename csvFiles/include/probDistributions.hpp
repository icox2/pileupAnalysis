//probDistributions.hpp
//this header file should be included when a probability distribution is needed or a random variable is needed.
#ifndef PROBDISTRIBUTIONS_HPP
#define PROBDISTRIBUTIONS_HPP

int Random(int low, int high);
bool Bernoulli(double probability);
int Triangular(double a, double b, double c);

#endif