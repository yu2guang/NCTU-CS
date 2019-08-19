float globalVar;
const int aint = 10;
const bool bbool = true;
const string sstr = "string";
const float rfloat = 1.23;
const double zo = 67E-1;
void april();
int banana(int left,int right);

int main() {

    int kok = 1;
    int loop_i = 100;
    print "start=";
    print loop_i;
    print "\n";
    if(3>5){
        loop_i = loop_i  + 200;
    }
    else{
        loop_i = loop_i + 800;
        if(3<=5){
            loop_i = loop_i  + 98;
        }
    }

    print "end=";
    print loop_i;
    print "\n";

    loop_i = 0;  
    print "loop:\n";
    while(loop_i<=10){
        loop_i = loop_i + 1;
        print loop_i;
        print " ";
    }
    print "\n";
   
    /*loop_i = 0;  
    int loop_j = 0; 
    while(loop_i<=10){
        print loop_i;
        print " ";
        loop_j = 0;
        print "do while:\n";
        do{
            loop_j = loop_j + 1;
            print loop_j;
            print " "; 
        } while(loop_j<=5);
        print "\n";
        loop_i = loop_i + 1;
    }
    print "for loop:\n";
    loop_i = 4; 
    for(; loop_i<=10;loop_i = loop_i + 1){
        print loop_i;
        print " ";
    } 
    print "\n";
    print "end\n";*/

    april();

    globalVar = 456.89;
    print globalVar;
    print "\n";

    int baa = banana(65,23);
    print "banana=";
    //print baa;
    print banana(65,23);
    print "\n";

    int i, result,k;
    result = 0;
    i = 1*3+aint;
    k = -i;
    bool gh = bbool;

    print i*rfloat;
    print "\n";

    globalVar = 15.3;

    int a1 = 8;
    bool b1;
    float c1;
    double r1,r2,r3;

    r1 = 45E-1;
    r2 = 58E-1;
    r3 = 93E-1;

    r1 = i;

    print "double=";
    print zo;
    print "\ndouble2=";
    print r1;

    print "\nint=";
    read a1;
    print "\nbool=";
    read b1;
    print "\nfloat=";
    read c1;
    print "\ndouble=";
    read r1;

    print "hey\n";
    print 12;
    print "\n";
    print 12E+2;
    print "\n";
    print 115.23;
    print "\n";
    print i;
    print "\n";
    print gh;
    print "\n";
    print k;
    print "\n";
    print "---const\n";
    print aint;
    print "\n";
    print bbool;
    print "\n";
    print sstr;
    print "\n";
    print rfloat;
    print "\n";
    print "---read\n";
    print a1;
    print "\n";
    print b1;
    print "\n";
    print c1;
    print "\n";
    print r1;
    print "\n";
    
    return 0;
}

void april(){
    print "pril";
    print "\n";
}

int banana(int left,int right){

    int value = 200;
    if(left<=right) {
        value = 1000899;
    }
    else{
        value = 345678;
    }

    return value;
}
