//
// Created by kubkm on 09.05.2022.

// Code taken from stack overflow, https://stackoverflow.com/questions/6852055/how-can-i-modify-pixels-using-sdl

//

#include "SdlUtils.h"


void SdlUtils::PutPixel32_nolock(SDL_Surface * surface, int x, int y, Uint32 color){
    auto * pixel = (Uint8*)surface->pixels;
    pixel += (y * surface->pitch) + (x * sizeof(Uint32));
    *((Uint32*)pixel) = color;
}

void SdlUtils::PutPixel24_nolock(SDL_Surface * surface, int x, int y, Uint32 color){
    auto * pixel = (Uint8*)surface->pixels;
    pixel += (y * surface->pitch) + (x * sizeof(Uint8) * 3);
#if SDL_BYTEORDER == SDL_BIG_ENDIAN
    pixel[0] = (color >> 24) & 0xFF;
    pixel[1] = (color >> 16) & 0xFF;
    pixel[2] = (color >> 8) & 0xFF;
#else
    pixel[0] = color & 0xFF;
    pixel[1] = (color >> 8) & 0xFF;
    pixel[2] = (color >> 16) & 0xFF;
#endif
}

void SdlUtils::PutPixel16_nolock(SDL_Surface * surface, int x, int y, Uint32 color){
    auto * pixel = (Uint8*)surface->pixels;
    pixel += (y * surface->pitch) + (x * sizeof(Uint16));
    *((Uint16*)pixel) = color & 0xFFFF;
}

void SdlUtils::PutPixel8_nolock(SDL_Surface * surface, int x, int y, Uint32 color){
    auto * pixel = (Uint8*)surface->pixels;
    pixel += (y * surface->pitch) + (x * sizeof(Uint8));
    *pixel = color & 0xFF;
}

void SdlUtils::PutPixel32(SDL_Surface * surface, int x, int y, Uint32 color){
    if( SDL_MUSTLOCK(surface) )
        SDL_LockSurface(surface);
    PutPixel32_nolock(surface, x, y, color);
    if( SDL_MUSTLOCK(surface) )
        SDL_UnlockSurface(surface);
}

void SdlUtils::PutPixel24(SDL_Surface * surface, int x, int y, Uint32 color){
    if( SDL_MUSTLOCK(surface) )
        SDL_LockSurface(surface);
    PutPixel24_nolock(surface, x, y, color);
    if( SDL_MUSTLOCK(surface) )
        SDL_LockSurface(surface);
}

void SdlUtils::PutPixel16(SDL_Surface * surface, int x, int y, Uint32 color){
    if( SDL_MUSTLOCK(surface) )
        SDL_LockSurface(surface);
    PutPixel16_nolock(surface, x, y, color);
    if( SDL_MUSTLOCK(surface) )
        SDL_UnlockSurface(surface);
}

void SdlUtils::PutPixel8(SDL_Surface * surface, int x, int y, Uint32 color){
    if( SDL_MUSTLOCK(surface) )
        SDL_LockSurface(surface);
    PutPixel8_nolock(surface, x, y, color);
    if( SDL_MUSTLOCK(surface) )
        SDL_UnlockSurface(surface);
}
