#include <iostream>

using namespace std;

int main() {
    
    int value = 4000000;
    
    // 4000000 = (4Y+5)*X
    for(int i=2; i<=2000000; i++){
        if(value%i==0){
            if(((value/i-5)%4==0)){
                cout << i << " " << (value/i-5)/4 << endl;
            }
        }
    }    
    return 0;
}
