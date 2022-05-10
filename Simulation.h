////
//// Created by kubkm on 09.05.2022.
////
//
////TODO: temporary class, its functionality should be implemented in Visualisation
//
//#ifndef NEURALNET_SIMULATION_H
//#define NEURALNET_SIMULATION_H
//
//#include <iostream>
//
////#include "SDL2.framework/Headers/SDL.h"
////#include "SDL2.framework/Headers/SDL_main.h"
//
//#include "Network_lib/Sequential.h"
//#include "Network_lib/Layers/DenseLayer.h"
//#include "Network_lib/Layers/OutputLayer.h"
//
//using namespace std;
//
//
//class Simulation {
//public:
//    void run();
//    void init();
//    void end();
//
//private:
//    void update();
//    void draw();
//
//    void drawNetwork(Sequential * network);
//
//    SDL_Window* window;
//    SDL_Surface* screenSurface;
//    SDL_Renderer* grenderer;
//    const int SCREEN_WIDTH = 640;
//    const int SCREEN_HEIGHT = 480;
//
//    bool running = true;
//    int delay = 1000;
//    SDL_Event e;
//
//    Sequential * network;
//
//    vector<Tensor> * Ys;
//    vector<Tensor> * Xs;
//
//
//    void drawLayer(int posX, Layer * layer);
//
//    void drawLayerConnections(int posX, Layer *layer, int nextLayerSize, int nextLayerPosX);
//};
//
//
//#endif //NEURALNET_SIMULATION_H
