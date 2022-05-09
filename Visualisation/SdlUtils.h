//
// Created by kubkm on 09.05.2022.
//

#ifndef NEURALNET_SDLUTILS_H
#define NEURALNET_SDLUTILS_H

#include "../SDL2/x86_64-w64-mingw32/include/SDL2/SDL_surface.h"


class SdlUtils {

public:

    void PutPixel32_nolock(SDL_Surface * surface, int x, int y, Uint32 color);
    void PutPixel24_nolock(SDL_Surface * surface, int x, int y, Uint32 color);
    void PutPixel16_nolock(SDL_Surface * surface, int x, int y, Uint32 color);
    void PutPixel8_nolock(SDL_Surface * surface, int x, int y, Uint32 color);

    void PutPixel32(SDL_Surface * surface, int x, int y, Uint32 color);
    void PutPixel24(SDL_Surface * surface, int x, int y, Uint32 color);
    void PutPixel16(SDL_Surface * surface, int x, int y, Uint32 color);
    void PutPixel8(SDL_Surface * surface, int x, int y, Uint32 color);

};


#endif //NEURALNET_SDLUTILS_H
