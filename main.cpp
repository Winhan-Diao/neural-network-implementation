#include <fstream>
#include "network.hpp"
#include "mnist.hpp"
#include "activation_functions.hpp"
#include "loss_functions.hpp"
#include "utils.hpp"
#include <float.h>
#include "samples.hpp"

using namespace std::literals;

int main() {
    // _controlfp_s(NULL, 0, _EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW);
    mnist();
}

