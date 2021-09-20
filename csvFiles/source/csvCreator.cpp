//csvCreator.cpp
#include <iostream>
#include <fstream>
#include <cmath>
#include <utility>
#include "probDistributions.hpp"
using namespace std;

double pileupCalc(double t, double phase, double expon);
double const risingConst = 48.5; //these times are in ns, like the time scale
double const decayConst = 69.3;
int const range = 200;
//int const minPhase = 10;
double const jitter = 50.;
double const phase0 = 180; //setting all initial pulses to be at 180 ns
pair<int,int> ampRange = make_pair(500,10000);
pair<int,int> expRange = make_pair(11, 18);

int main(int argc, char **argv){
    if(argc<3){
        cout<<"Need to include number of files then minimum phase"<<endl;
        return 0;
    }
    int numFiles=atoi(argv[1]);
    int minPhase=atoi(argv[2]);
    int maxTime=1000, pileup; //maxTime in ns
    ofstream fLarge, fLabels;
    fLarge.open("traceList.csv");
    fLabels.open("traceLabels.csv");
    for(int i=0;i<numFiles;i++){
        if(i%1000==0) cout<<"\r"<<i<<flush;
        int x=0, y=0;
        ofstream fout;
        float phase =0., scale=0., amplitude=0.;
        float expon=0.;
        string prefix = "files/pileupRand_", ending=".csv", fName;
        //pileup = rand()%2+(i>100);
        pileup = int(Bernoulli(0.55))+(i>100); //Uses the Bernoulli distribution to place a weight on the number of pileups created
        fName = prefix+to_string(i)+ending;
        phase = Triangular((double)minPhase,35.0,(double)(minPhase+range+50));
        //phase = rand()%range + minPhase;
        scale = 0.5+(rand()%100)/200.;
        amplitude = double(rand()%(ampRange.second-ampRange.first)+ampRange.first);
        expon = double(rand()%(expRange.second-expRange.first)+expRange.first)/10.0;
        //fout.open(fName.c_str());
        for(int j=0;j<maxTime/2;j++){
            x = 2*j;
            if(pileup==2)
                y = amplitude*pileupCalc(x,phase0,expon)+scale*amplitude*pileupCalc(x,phase0+phase,expon)+jitter*(rand()%100)/100.;
            else if(pileup==1)
                y = amplitude*pileupCalc(x,phase0,expon)+jitter*(rand()%100)/100.;
            else
                y = jitter*(rand()%100)/100.;
            //fout<<x<<","<<y<<endl;
            
            if(j==0) fLarge<<y;
            else
                fLarge<<","<<y;
        }
        //fout.close();
        fLarge<<endl;
        fLabels<<pileup<<endl;
    }
    fLarge.close();
    fLabels.close();
    return 0;
}

double pileupCalc(double t, double phase, double expon){
    double first=0., second=0.;
    //first = (1-exp(-1*(t)/risingConst))*exp(-1*t/decayConst);
    //second = pow(1+exp(-1*(t-phase)/risingConst),-1)*exp(-1*(t-phase)/(decayConst));
    if(t>phase)
        first = exp(-1*(t-phase)/decayConst)*(1-exp(-1*pow(t-phase,expon)/risingConst));
    //second = exp(-1*(t-20-phase)/decayConst)*(1-exp(-1*pow(t-20-phase,expon)/risingConst));
    return first;
}