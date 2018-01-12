#include <iostream>


using namespace std;





int main (int argc, char* argv[] ) {


    int xadj[] {0,2,4,7,11,14};
    unsigned  long nVtx = 6;


    
    unsigned long x1 = nVtx << 32;  
    unsigned long x2 = nVtx << (64 - 32);  

    cout <<" 6 << 32  "<< x1 << " -  " << x2 << endl;
}
