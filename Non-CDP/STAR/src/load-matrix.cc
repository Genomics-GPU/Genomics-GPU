#include <assert.h>
#include "load-matrix.h"

using namespace std;

char* idx_mat[25];
int val_mat[25][25];

// trim from start
static inline string &ltrim(string &s) {
    s.erase(s.begin(), find_if(s.begin(), s.end(), not1(ptr_fun<int, int>(isspace))));
    return s;
}

// trim from end
static inline string &rtrim(string &s) {
    s.erase(find_if(s.rbegin(), s.rend(), not1(ptr_fun<int, int>(isspace))).base(), s.end());
    return s;
}

// trim from both ends
static inline string &trim(string &s) {
    return ltrim(rtrim(s));
}

// determine if a string is a number
bool is_number(const string& s) {
    // negative
    if (s.size() == 2) return true;
    return !s.empty() && find_if(s.begin(), s.end(), [](char c) { return !isdigit(c); }) == s.end();
}


// load matrix
int load_matrix(const char *path) {
    string buff, line;
    string delimiter = " ";

    ifstream file;
    file.open(path);
    assert(file);

    for (int i = 0; i < 25; i++) {
        idx_mat[i] = (char *)"";
        for (int j = 0; j < 25; j++)
            val_mat[i][j] = 0;
    }

    int row = 0;

    while (getline(file, buff)) {
        if (buff.empty() || buff[0] == '#' || buff[0] == '*' || buff[0] == ' ') {
            continue;
        }
        else {
            // trim both spaces of head and tail
            line = trim(buff);

            size_t pos = 0;
            string token;
            int col = 0;

            while ((pos = line.find(delimiter)) != string::npos) {
                token = line.substr(0, pos);
                // filter double space
                if (token != "") {
                    // determine if a string is a number
                    if (is_number(token)) {
                        val_mat[row][col] = atoi(token.c_str());
                        //cout << "int:" << token << "," << val_mat[row][col] << endl;
                        col++;
                    }
                    else {
                        idx_mat[row] = &token[0u];
                        //cout << "char:" << token << "," << idx_mat[row] << endl;
                    }
                }
                line.erase(0, pos + delimiter.length());
            }
            row++;
        }
    }
    return 0;
}

// searchScore value by row_n and col_n
int searchScore(char x, char y) {
    for (int i = 0; i < 25; i++) {
        if (x == *idx_mat[i] && isalpha(*idx_mat[i])) {
            for (int j = 0; j < 25; j++) {
                if (y == *idx_mat[j] && isalpha(*idx_mat[j]))
                    return val_mat[i][j];
            }
        }
    }
    return 0;
}








