#include <iostream>

#include "Tensor.h"
#include "Layers/DenseLayer.h"
#include "Sequential.h"
#include "ModelLoader.h"
#include "Layers/OutputLayer.h"

#include <SDL.h>
#include <SDL_main.h>
#include "Simulation.h"

using namespace std;


void runSimulationStep(){

}

int main(int argv, char** args) {


    Simulation sim;

    sim.init();
    sim.run();

    return 0;
}
