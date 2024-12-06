#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <stdint.h>

using namespace std;
int main() {
    unsigned char c = 0;
    c=256;
    printf("c = %d\n", c);
    c=min(max(int(256), 0), 255);
    printf("c = %d\n", c);
}