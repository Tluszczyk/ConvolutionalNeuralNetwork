//
// Created by kubkm on 09.05.2022.
//

#include "Simulation.h"

void Simulation::init() {
    if( SDL_Init( SDL_INIT_VIDEO ) < 0 )
    {
        printf( "SDL could not initialize! SDL_Error: %s\n", SDL_GetError() );
        exit(-1);
    }

    window = SDL_CreateWindow( "Neural Network", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN );
    if( window == nullptr )
    {
        printf( "Window could not be created! SDL_Error: %s\n", SDL_GetError() );
        exit(-2);
    }

    screenSurface = SDL_GetWindowSurface( window );
    grenderer =  SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);


    network = new Sequential({
                                          new DenseLayer({2}, "sig", "Dense input"),
                                          new DenseLayer({4}, "sig", "Idk"),
                                          new DenseLayer({8}, "sig", "Idk"),
                                          //new DenseLayer({8}, "sig", "Idk"),
                                          new DenseLayer({4}, "sig", "Idk"),
                                          new OutputLayer({2}, "id", "Dense output"),
                                  });

    network->compile(0.8);

    //ModelLoader::loadFromFile("C:/Users/kubkm/CLionProjects/ConvolutionalNeuralNetwork/xor_model.mdl");

    vector<Tensor> X1 = {Tensor({2}, {0,1})};
    vector<Tensor> X2 = {Tensor({2}, {1,0})};
    vector<Tensor> X3 = {Tensor({2}, {1,1})};
    vector<Tensor> X4 = {Tensor({2}, {0,0})};

    vector<Tensor> Y1 = {Tensor({2}, {0,1})};
    vector<Tensor> Y2 = {Tensor({2}, {0,1})};
    vector<Tensor> Y3 = {Tensor({2}, {1,1})};
    vector<Tensor> Y4 = {Tensor({2}, {0,0})};

    Xs = new vector<Tensor>[4] {X1, X2, X3, X4};
    Ys = new vector<Tensor>[4] {Y1, Y2, Y3, Y4};
}

void Simulation::drawNetwork(Sequential * network){
    for (int i = 0; i < network->layers.size() - 1; i++){
        drawLayerConnections(((SCREEN_WIDTH - 150) * i / network->layers.size()) + 75, network->layers[i],
                  network->layers[i+1]->getShape()[0], ((SCREEN_WIDTH - 150) * (i + 1) / network->layers.size()) + 75);
    }
    for (int i = 0; i < network->layers.size(); i++){
        drawLayer(((SCREEN_WIDTH - 150) * i / network->layers.size()) + 75, network->layers[i]);
    }
}

void Simulation::drawLayer(int posX, Layer * layer){
    Tensor activations = layer->getActivations();
    vector<double> activationData = activations.getData();
    for (int i = 0; i < activationData.size(); i++){
        SDL_Rect fillRect = { posX - 10, static_cast<int>(((SCREEN_HEIGHT - 200) * i / activationData.size())) + 90, 20, 20 };
        SDL_SetRenderDrawColor(grenderer, 0, 0, activationData[i] * 250 + 5, 255 );
        if (activationData[i] > 1){
            cout<<"wieksze"<<endl;
        }
        SDL_RenderFillRect(grenderer, &fillRect );
    }
}

void Simulation::drawLayerConnections(int posX, Layer * layer, int nextLayerSize, int nextLayerPosX){
    if (nextLayerSize > 0) {

        Tensor weights = layer->getWeights();

        for (int i = 0; i < weights.getShape()[0]; i++) {
            for (int j = 0; j < weights.getShape()[1]; j++) {
                if (weights[{i, j}] > 0){
                    SDL_SetRenderDrawColor(grenderer, 0, weights[{i, j}]*-250, 0, 255 );
                } else{
                    if (weights[{i, j}] > 1){
                        cout<<"a problem"<<endl;
                    }
                    SDL_SetRenderDrawColor(grenderer, weights[{i, j}]*250, 0, 0, 255 );
                }
                SDL_RenderDrawLine(grenderer,
                                   posX, static_cast<int>(((SCREEN_HEIGHT - 200) * i / weights.getShape()[0])) + 100,
                                   nextLayerPosX, static_cast<int>(((SCREEN_HEIGHT - 200) * j / weights.getShape()[1])) + 100);
            }
        }
    }
}

void Simulation::run() {
    while (running){
        update();
        draw();
        SDL_Delay( delay );
    }
    end();
}

void Simulation::update() {


    while( SDL_PollEvent( &e ) != 0 )
    {
        //User requests quit
        if( e.type == SDL_QUIT )
        {
            running = false;
        }
    }

    for (int i = 0; i < 100; i++) {
        int tmp = rand() % 4;
        network->analyzeBatch(Xs[tmp], Ys[tmp]);
    }


}

void Simulation::draw() {
    SDL_SetRenderDrawColor(grenderer, 255, 255, 255, 255 );
    SDL_RenderClear(grenderer );
    drawNetwork(network);
    SDL_RenderPresent(grenderer);
}

void Simulation::end() {
    SDL_DestroyWindow( window );
    SDL_DestroyRenderer(grenderer );
    window = nullptr;
    grenderer = nullptr;
    SDL_Quit();
}



