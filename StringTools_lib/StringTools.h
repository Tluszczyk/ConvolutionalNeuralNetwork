//
// Created by Filip Tłuszcz on 09.05.2022.
//

#ifndef NEURALNET_STRINGTOOLS_H
#define NEURALNET_STRINGTOOLS_H

#include <string>
#include <vector>

using namespace std;

class StringTools {
public:
    static vector<string> split (string s, string delimiter);
    static string ltrim(string s);
    static string rtrim(string s);
    static string trim(string s);
};


#endif //NEURALNET_STRINGTOOLS_H
