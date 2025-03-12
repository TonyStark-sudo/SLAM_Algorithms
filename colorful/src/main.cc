#include "projector.h"
#include <iostream>

int main(int argc, char** argv) {
    Projector projector(argv[1]);
    projector.ProjectColorful();
    return 0;
}
