#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <unistd.h>
#include "mpi.h"
#include <pthread.h>

#define ITER 1000
#define COUNT_TASKS 10000
#define IS_FINISH -1
#define NO_TASK -1

typedef struct Context{
    int globalRes;
    int rank, size;
    int *tasks;
    int countRemainingTasks;
    int currentCountTasks;
    pthread_mutex_t mutex;
} Context;

void genetateTasks(Context* context, int iterCounter){
    for(int i = 0; i < COUNT_TASKS; ++i){
        context->tasks[i] = abs(context->rank - (iterCounter % context->size));
    }
}

void executeTask(Context* context, int weight){
    int res = 0;
    for(int i = 0; i < weight; ++i)
        res += sqrt(i);
    context->globalRes += res;
}

void executeTasks(Context* context){
    pthread_mutex_lock(&context->mutex);
    int i = context->currentCountTasks - context->countRemainingTasks;
    pthread_mutex_unlock(&context->mutex);
    while(true){
        pthread_mutex_lock(&context->mutex);
        if(i >= context->currentCountTasks){
            pthread_mutex_unlock(&context->mutex);
            break;
        }
        context->countRemainingTasks--;
        pthread_mutex_unlock(&context->mutex);
        executeTask(context, context->tasks[i]);
        i++;
    }
}

void requestTasks(Context* context){
    MPI_Status status;
    for(int i = context->rank + 1; i < context->size; ++i){
        MPI_Send(&context->rank, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        printf("Process %d: send request tasks to %d process\n", context->rank, i);
        int recvMessage;
        MPI_Recv(&recvMessage, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if(recvMessage != NO_TASK){
            int task = recvMessage;
            printf("Process %d: recieve 1 task from %d process\n", context->rank, i);
            pthread_mutex_lock(&context->mutex);
            context->tasks[context->currentCountTasks] = task;
            context->currentCountTasks++;
            context->countRemainingTasks++;
            pthread_mutex_unlock(&context->mutex);
            executeTasks(context);
            i--;
        }
        else{
            printf("Process %d: no tasks from %d process\n", context->rank, i);
        }
    }
}

void *execute(void *cont){
    Context *context = ((Context*)cont);
    int iterCounter = 0;
    while(iterCounter < ITER){
        printf("Process %d: ITER: %d\n", context->rank, iterCounter);
        genetateTasks(context, iterCounter);
        MPI_Barrier(MPI_COMM_WORLD);
        printf("Process %d: generated tasks\n", context->rank);
        executeTasks(context);
        printf("Process %d: executed %d tasks(remain %d)\n", context->rank, context->currentCountTasks, context->countRemainingTasks);
        requestTasks(context);
        MPI_Barrier(MPI_COMM_WORLD);
        pthread_mutex_lock(&context->mutex);
        context->currentCountTasks = COUNT_TASKS;
        context->countRemainingTasks = COUNT_TASKS;
        pthread_mutex_unlock(&context->mutex);
        iterCounter++;
    }
    printf("Process %d: finish\n", context->rank);
    int messageIsFinish = IS_FINISH;
    int sendRank = (context->rank + 1) % context->size;
    MPI_Send(&messageIsFinish, 1, MPI_INT, sendRank, 0, MPI_COMM_WORLD);
    return NULL;
}

void *waitRequests(void *cont){
    Context *context = ((Context*)cont);
    int recvRank, recvMessage;
    MPI_Status status;
    MPI_Barrier(MPI_COMM_WORLD);
    while(true){
        printf("Process %d: wait for request...\n", context->rank);
        MPI_Recv(&recvMessage, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        if(recvMessage == IS_FINISH)
            break;
        recvRank = recvMessage;
        printf("Process %d: recieve request from %d process\n", context->rank, recvRank);
        pthread_mutex_lock(&context->mutex);
        if(context->countRemainingTasks > 1){
            context->currentCountTasks--;
            context->countRemainingTasks--;
            pthread_mutex_unlock(&context->mutex);
            MPI_Send(context->tasks + context->currentCountTasks, 1, MPI_INT, recvRank, 1, MPI_COMM_WORLD);
            printf("Process %d: send 1 task to %d process\n", context->rank, recvRank);
        }
        else{
            pthread_mutex_unlock(&context->mutex);
            int messageNoTask = NO_TASK;
            MPI_Send(&messageNoTask, 1, MPI_INT, recvRank, 1, MPI_COMM_WORLD);
            printf("Process %d: no tasks to share for %d process\n", context->rank, recvRank);
        }
    }
    return NULL;
}
 
int main(int argc, char* argv[]){
    Context context;
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    if (provided != MPI_THREAD_MULTIPLE){ 
        MPI_Finalize(); 
        return 0; 
    }
    
    MPI_Comm_size(MPI_COMM_WORLD, &context.size);
    MPI_Comm_rank(MPI_COMM_WORLD, &context.rank);

    pthread_attr_t attrs;
    pthread_attr_init(&attrs);

    pthread_mutex_init(&context.mutex, NULL);

    pthread_t threads[2];

    pthread_attr_setdetachstate(&attrs, PTHREAD_CREATE_JOINABLE);

    context.tasks = malloc(context.size * COUNT_TASKS * sizeof(int));
    context.countRemainingTasks = COUNT_TASKS;
    context.currentCountTasks = COUNT_TASKS;
    context.globalRes = 0;

    double timeStart = MPI_Wtime();
    pthread_create(&threads[1], &attrs, waitRequests, (void*)&context);
    pthread_create(&threads[0], &attrs, execute, (void*)&context);

    pthread_join(threads[0], NULL);
    pthread_join(threads[1], NULL);
    pthread_attr_destroy(&attrs);    
    pthread_mutex_destroy(&context.mutex);

    free(context.tasks);

    double timeFinish = MPI_Wtime();
    if(context.rank == 0)
        printf("Time: %f\n", timeFinish - timeStart);

    MPI_Finalize();
    return 0;
}
