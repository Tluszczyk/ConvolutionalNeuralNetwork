//
// Created by Filip Tłuszcz on 25.03.2022.
//

#include "Sequential.h"

Tensor Sequential::feed(Tensor inputTensor) {
    /* feed a single tensor to a model,
     * and get the activations of the last layer
     * */
    return Tensor({2}, {1,2});
}

void Sequential::backpropagate(Tensor gradient) {
    /* adjust all the adjustable parameters
     * */

}
