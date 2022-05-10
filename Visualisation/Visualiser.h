//
// Created by kubkm on 09.05.2022.
//

#ifndef NEURALNET_VISUALISER_H
#define NEURALNET_VISUALISER_H

#include <SDL.h>
#include <stdio.h>


class Visualiser {

    //Starts up SDL and creates window
    bool init();

//Loads media
    bool loadMedia();

//Frees media and shuts down SDL
    void close();

};


#endif //NEURALNET_VISUALISER_H
