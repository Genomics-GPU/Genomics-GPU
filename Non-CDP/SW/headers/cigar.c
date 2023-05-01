
#include "cigar.h"


char * readFragment(FILE * file, size_t size){

	char * str;
	int c;
	size_t length = 0;

	str = realloc(NULL, sizeof(char) * size);
	if(!str)return str;
	while( (c = fgetc(file)) != EOF && c != '\n'){
		str[length++] = c;
		if(length == size){
			str = realloc(str, sizeof(char) * (size+=size));
			if(!str)return str;
		}
	}
	str[length++] = '\0';

	return realloc(str, sizeof(char) * length);


}



char * compressCigar(char * uncompressedCigar){

	int length = strlen(uncompressedCigar);
	char * compressedCigar = calloc(sizeof(char), length);
	int start = length-1;

	while(start > 0){
		
		if(uncompressedCigar[start] == uncompressedCigar[start-1]){
			int count = 1;

			while(uncompressedCigar[start] == uncompressedCigar[start-1]){

				count++;
				start--;

			}
			
			char buf[15];
			sprintf(buf, "%d%c", count, uncompressedCigar[start]);
			strcat(compressedCigar, buf);

		}else{
			char * buf = calloc(sizeof(char), 1);
			*buf = uncompressedCigar[start];
			strcat(compressedCigar, buf);
			free(buf);	
		}


		start--;
	}


	return compressedCigar;
}
