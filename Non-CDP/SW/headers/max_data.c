#include "max_data.h"

linkedList * initList(){

	linkedList * list = malloc(sizeof(linkedList));
	list->head = NULL;
	list->tail = NULL;
	return list;
}

void printEntry(maxEntry * entry){

    printf("(x = %d; y = %d)\n", entry->x, entry->y);


}
maxEntry * append(linkedList * list, int x, int y){
	

	maxEntry * entry = malloc(sizeof(maxEntry));
	entry->x = x;
	entry->y = y;

	if(list->size == 0){
	    list->head = entry;
	    list->tail = entry;
	}else{
	    list->tail->next = entry;
	    list->tail = entry;
	}
	list->size++;
	
	entry->next = NULL;

	return entry;
}

int clearList(linkedList * list){
	
	
	maxEntry * temp;

	while(list->head != NULL){
	    temp = list->head;
	    list->head = list->head->next;
	    free(temp);
	}
	list->tail = NULL;
	list->size = 0;

	return 0;
}

void freeList(linkedList * list){

	clearList(list);
	free(list);
}

