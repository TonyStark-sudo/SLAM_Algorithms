#include <iostream>

using namespace std;

class Myclass {
public:
    Myclass(int x = 3) : _x(x) {}

    int operator() (const int n) const {
        return n * _x;
    }

private:
    int _x;
};

int main(int argc, char *argv[])
{   
    // Myclass myclass(2);
    Myclass myclass;
    
    // !!!!!知道这里打印数字是多少才真正明白了运算符重载
    cout << "Overload operator (): " << myclass(1) << endl;
    cout << "Overload operator (): " << myclass(2) << endl;
    cout << "Overload operator (): " << myclass(5) << endl;
    /* code */
    return 0;
}

