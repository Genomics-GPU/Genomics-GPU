#ifndef CIGAR__H
#define CIGAR__H
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

char * readFragment(FILE * file, size_t size);
char * compressCigar(char * uncompressedCigar);
#endif

