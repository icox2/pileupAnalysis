//probDistributions.cpp
#include "probDistributions.hpp"
#include <cmath>
#include <utility>
#include <stdlib.h>

int Random(int low, int high){
    return rand()%(high-low)+low;
}

bool Bernoulli(double probability){
    double val = rand()/(double)RAND_MAX;
    if(val*probability>=0.5)
        return true;
    else
        return false;
}

int Triangular(double a, double b, double c){
    double U = rand()/(double)RAND_MAX;
    double F = (b-a)/(a-a);
    if(U<=F)
        return a+sqrt(U*(b-a)*(c-a));
    else    
        return b-sqrt((1-U)*(b-a)*(b-c));
}