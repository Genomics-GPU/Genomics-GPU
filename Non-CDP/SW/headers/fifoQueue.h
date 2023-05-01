#ifndef FIFOQUEUE__H
#define FIFOQUEUE__H


typedef struct queueEntry{
	int x;
	int y;
	struct queueEntry* next;
}fEntry;

typedef struct fifo_queue{
	fEntry * top;
	fEntry * last;
	int size;
}fQueue;

int isEmpty(fQueue* queue);



fQueue * initQueue(){

	fQueue* queue = malloc(sizeof(fQueue));
	queue->top = NULL;
	queue->last = NULL;
	queue->size = 0;
	return queue;

}

fEntry * addEntry(int x, int y, fQueue* queue){


	fEntry*  entry = malloc(sizeof(fEntry));
	entry->x = x;
	entry->y = y;
	entry->next = NULL;

	if(queue->size == 0){
		queue->top = entry;
		queue->last = entry;
		queue->size++;
		entry->next = NULL;	
	}else{
		queue->last->next = entry;
		queue->last = entry;
		queue->size++;
	}
	return entry;
}

fEntry*  pop(fQueue * queue){
	if(isEmpty(queue)){
		return NULL;
	}
       
	fEntry * t = queue->top;
	if(queue->top == queue->last){
		queue->size = 0;
		queue->top = NULL;
		queue->last = NULL;
		
	}else{
		queue->top = queue->top->next;
		queue->size--;
	}
	return t;

}
int isEmpty(fQueue* queue){
	return queue->size == 0;
}
int deleteEntry(fEntry * entry){
	free(entry);
	return 0;
}
int deleteQueue(fQueue * queue){

	if(isEmpty(queue)) free(queue);
	else{
	    for(int i = 0; i < queue->size; i++){
		fEntry * t = pop(queue);
	    	free(t);	
	    }
	}
	return 0;
}

#endif
