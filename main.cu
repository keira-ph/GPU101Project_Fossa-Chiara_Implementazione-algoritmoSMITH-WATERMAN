#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define S_LEN 512
#define N 1000
#define CHECK(call)                                                                       \
    {                                                                                     \
        const cudaError_t err = call;                                                     \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }

#define CHECK_KERNELCALL()                                                                \
    {                                                                                     \
        const cudaError_t err = cudaGetLastError();                                       \
        if (err != cudaSuccess)                                                           \
        {                                                                                 \
            printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(EXIT_FAILURE);                                                           \
        }                                                                                 \
    }


double get_time() // function to get the time of day in seconds
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int max4(int n1, int n2, int n3, int n4)
{
    int tmp1, tmp2;
    tmp1 = n1 > n2 ? n1 : n2;
    tmp2 = n3 > n4 ? n3 : n4;
    tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
    return tmp1;
}
__device__  int max4GPU(int n1, int n2, int n3, int n4)
{
    int tmp1, tmp2;
    tmp1 = n1 > n2 ? n1 : n2;
    tmp2 = n3 > n4 ? n3 : n4;
    tmp1 = tmp1 > tmp2 ? tmp1 : tmp2;
    return tmp1;
}
void backtrace(char *simple_rev_cigar, char **dir_mat, int i, int j, int max_cigar_len)
{
    int n;

    for (n = 0; n < max_cigar_len && dir_mat[i][j] != 0; n++)
    {
        int dir = dir_mat[i][j];
        if (dir == 1 || dir == 2)
        {
            i--;
            j--;
        }
        else if (dir == 3)
            i--;
        else if (dir == 4)
            j--;

        simple_rev_cigar[n] = dir;
    }
    if(dir_mat[i][j] == 0)
        simple_rev_cigar[n] = 0;

}
__device__ void backtraceGPU(char *simple_rev_cigar, char *dir_mat, int maxind, int max_cigar_len) {

    int tid = threadIdx.x;
    int row = blockIdx.x;
    __shared__ int countonnext,prec;
  __shared__ int  x, righe, colonne;
    __shared__ int n,stop;
    __shared__ int next[1024];

    n=0;
    stop=0;
    __shared__ int partenza;
    partenza=((maxind+1)%(513*513))-1;




    __syncthreads();

   while ( n<2*512 && stop==0)
    {

        if(tid==0){
            righe=(partenza+1)/513+1;
            colonne=(partenza+1)%513;
            if(righe*colonne>=1024){
                if(righe>31 && colonne>31){
                    righe=32;
                    colonne=32;
                }else if(righe<=31){
                    colonne=1024/righe;
                }else if(colonne<=31){
                    righe=1024/colonne;
                }
            }
        }

        __syncthreads();

        if(tid<righe*colonne){
            x=partenza-(tid/colonne)*513- tid%colonne;
            int dir = dir_mat[row*513*513+x];
            if (dir == 1 || dir == 2){
               if(((partenza-x+1)/513)<righe-1 && ((partenza-x)%513)<colonne-1)
                   next[tid]=tid + colonne + 1;
                else
                next[tid]=1024;

            }
            else if (dir == 3){
                if(((partenza-x)/(513))<righe-1)
                    next[tid]=tid+colonne;
                else
                     next[tid]=1025;
            }
            else if (dir == 4){
                if((partenza-x)%513<colonne-1)
                next[tid]=tid+1;
                else
                next[tid]=1026;
            }

            else if (dir == 0)
                next[tid]=-1;


        }

        __syncthreads();
        /*if(tid==0 && next[0]==0 ){

            printf("%d-", next[0]);
            }*/




        if(tid==0){
            countonnext=0;
            prec=0;
            /*if(n==0 && row==0)
                printf("%d ",dir_mat[partenza]);*/
             while( countonnext >=0 && countonnext<(righe*colonne)){
                 simple_rev_cigar[n] = dir_mat[row*513*513+partenza-(countonnext/colonne)*513-(countonnext%colonne)];
                 prec=countonnext;
                 countonnext=next[countonnext];
                 n++;
             }
            if (countonnext!=-1 && countonnext!=1024 && countonnext!=1025 && countonnext!=1026){
                printf("ERR%d%d%dn", countonnext, prec,n);

            }

             if (countonnext==1024)
                 partenza=partenza-((prec)/colonne)*513-(prec)%colonne-514;
             else if (countonnext==1025)
                 partenza=partenza-((prec)/colonne)*513-(prec)%colonne-513;
             else if (countonnext==1026)
                 partenza=partenza-((prec)/colonne)*513-(prec)%colonne-1;
             else if (countonnext==-1){
                 stop=1;
                 simple_rev_cigar[n]=0;

             }
        }


        __syncthreads();

     }

    /*if(row==33 && tid==1023)
        printf("R%d-T%d ", row, tid);*/

}

__global__ void inplinGpu(char* gquery, char* greference, char* gsimple_rev_cigar, int *gres, int* sc_mat, char* dir_mat) {

    int tid = threadIdx.x;
    int row = blockIdx.x;

    /*if(row==999 && tid==1023)
        printf("R%d-T%d ", row, tid);*/
   __shared__ int max[1025];
    __shared__ int posizioni[1025];
    if(tid<1024)
      max[tid]=-2;
    if(tid==0){
       max[1024]=-2;

    }






    // initialize the scoring matrix and direction matrix to 0
    if(tid<1024){


        for (int j = 0; j < 257; j++)
        {
            sc_mat[(row*513*513)+(tid * 257 + j)] = 0;

            dir_mat[(row*513*513)+(tid * 257 + j)] = 0;
        }
        if(tid==0)
            sc_mat[(row*513*513)+((S_LEN+1)*(S_LEN+1)-1)] = 0;
        if(tid==1023)
            dir_mat[(row*513*513)+((S_LEN+1)*(S_LEN+1)-1)] = 0;
    }



    __syncthreads();


    // compute the alignment
    for (int n = 2; n<= 1024; n++) {
        if (tid>0 && tid < n && tid<=511 && (n-tid)<=511) {
            // compare the sequences characters
            int comparison = (gquery[(row)*S_LEN+(tid-1)] == greference[(row)*S_LEN+(n-tid-1)]) ? 1 : -1;
            // compute the cell knowing the comparison result
            int x=(row*513*513)+((tid)*513+(n-tid));
            int tmp = max4GPU(sc_mat[x-514] + comparison, sc_mat[x-513] -2, sc_mat[x-1] -2, 0);
            char dir;
            if (tmp == (sc_mat[x-514] + comparison))
                dir = comparison == 1 ? 1 : 2;
            else if (tmp == (sc_mat[x-513] -2))
                dir = 3;
            else if (tmp == (sc_mat[x-1] -2))
                dir = 4;
            else
                dir = 0;
            dir_mat[x] = dir;
            sc_mat[x] = tmp;

        }
        __syncthreads();
    }

   /* if(tid==0 && row==333){
        for(int i=0; i<100; i++)
            printf("GS%d-D%d ",sc_mat[333*513*513+333*513+i], dir_mat[333*513*513+333*513+i]);
    }*/

    int j;
    if(tid<1024){
        if(tid==0){
            max[1024]=sc_mat[(row*513*513)+(1024 * 257 )];
            posizioni[1024]=(row*513*513)+(1024 * 257 );

        }
        for ( j = 0; j < 257; j++)
        {
            if(sc_mat[(row*513*513)+(tid * 257 + j)]>max[tid]){
                max[tid]=sc_mat[(row*513*513)+(tid * 257 + j)];
                posizioni[tid]=(row*513*513)+(tid * 257 + j);
            }
        }


    }
    __syncthreads();


    if(tid==0){
        for(j=0; j<=1024; j++){
            if(max[j]>max[0]){
                max[0]=max[j];
                posizioni[0]=posizioni[j];
            }
        }



    }/*
    if(tid==0){
        for(int t=0; t<513*513;  t++){
            if(gres[row]<sc_mat[(row*513*513)+t])
                gres[row]=sc_mat[(row*513*513)+t];
        }
    }*/
    gres[row] = max[0];


    __syncthreads();


    backtraceGPU(&gsimple_rev_cigar[row*512*2], dir_mat, posizioni[0], 512 * 2);


}

int main(int argc, char *argv[])
{
    srand(time(NULL));

    char alphabet[5] = {'A', 'C', 'G', 'T', 'N'};

    int ins = -2, del = -2, match = 1, mismatch = -1; // penalties

    char **query = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        query[i] = (char *)malloc(S_LEN * sizeof(char));

    char **reference = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        reference[i] = (char *)malloc(S_LEN * sizeof(char));

    int **sc_mat = (int **)malloc((S_LEN + 1) * sizeof(int *));
    for (int i = 0; i < (S_LEN + 1); i++)
        sc_mat[i] = (int *)malloc((S_LEN + 1) * sizeof(int));
    char **dir_mat = (char **)malloc((S_LEN + 1) * sizeof(char *));
    for (int i = 0; i < (S_LEN + 1); i++)
        dir_mat[i] = (char *)malloc((S_LEN + 1) * sizeof(char));

    int *res = (int *)malloc(N * sizeof(int));
    char **simple_rev_cigar = (char **)malloc(N * sizeof(char *));
    for (int i = 0; i < N; i++)
        simple_rev_cigar[i] = (char *)malloc(S_LEN * 2 * sizeof(char));

    char *Q = (char *)malloc(N*S_LEN * sizeof(char));
    char *R = (char *)malloc(N*S_LEN * sizeof(char));
    int *RES = (int *)malloc(N * sizeof(int));
    char *SRC = (char *)malloc(N*2*S_LEN * sizeof(char));


    // randomly generate sequences
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < S_LEN; j++)
        {
            query[i][j] = alphabet[rand() % 5];

            Q[i*S_LEN+j]=query[i][j];
            reference[i][j] = alphabet[rand() % 5];
            R[i*S_LEN+j]=reference[i][j];
        }
    }

    double start_cpu = get_time();

    for (int n = 0; n < N; n++)
    {
        int max = ins; // in sw all scores of the alignment are >= 0, so this will be for sure changed
       int maxi, maxj;
        // initialize the scoring matrix and direction matrix to 0
        for (int i = 0; i < S_LEN + 1; i++)
        {
            for (int j = 0; j < S_LEN + 1; j++)
            {
                sc_mat[i][j] = 0;
                dir_mat[i][j] = 0;
            }
        }
        // compute the alignment
        for (int i = 1; i < S_LEN; i++)
        {
            for (int j = 1; j < S_LEN; j++)
            {
                // compare the sequences characters
                int comparison = (query[n][i - 1] == reference[n][j - 1]) ? match : mismatch;
                // compute the cell knowing the comparison result
                int tmp = max4(sc_mat[i - 1][j - 1] + comparison, sc_mat[i - 1][j] + del, sc_mat[i][j - 1] + ins, 0);
                char dir;

                if (tmp == (sc_mat[i - 1][j - 1] + comparison))
                    dir = comparison == match ? 1 : 2;
                else if (tmp == (sc_mat[i - 1][j] + del))
                    dir = 3;
                else if (tmp == (sc_mat[i][j - 1] + ins))
                    dir = 4;
                else
                    dir = 0;

                dir_mat[i][j] = dir;
                sc_mat[i][j] = tmp;

                if (tmp > max)
                {
                    max = tmp;
                    maxi = i;
                    maxj = j;
                }
            }
        }

        res[n] = sc_mat[maxi][maxj];


        backtrace(simple_rev_cigar[n], dir_mat, maxi, maxj, S_LEN * 2);


    }
    /*printf("%d",simple_rev_cigar[0][0]);*/
    double end_cpu = get_time();
    printf("SW Time CPU: %.10lf\n", end_cpu - start_cpu);
    char *gquery, *greference,*gdir ,*gsimple_rev_cigar;
    int *gsc, *gres;

    CHECK(cudaMalloc(&gquery, N*S_LEN * sizeof(char)));
    CHECK(cudaMalloc(&greference, N*S_LEN * sizeof(char)));

    CHECK(cudaMalloc(&gsimple_rev_cigar, 2*N*S_LEN * sizeof(char)));
    CHECK(cudaMalloc(&gres, N * sizeof(int)));

    CHECK(cudaMalloc(&gsc, N*(S_LEN + 1) * (S_LEN + 1) * sizeof(int)));
    CHECK(cudaMalloc(&gdir, N*(S_LEN + 1) * (S_LEN + 1) * sizeof(char)));



    CHECK(cudaMemcpy(gquery, Q, N*S_LEN * sizeof(char), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(greference, R, N*S_LEN * sizeof(char), cudaMemcpyHostToDevice));

    dim3 blocksPerGrid(1000, 1, 1);
    dim3 threadsPerBlock(1024, 1, 1);

    double start_gpu = get_time();

    inplinGpu<<<blocksPerGrid, threadsPerBlock>>>(gquery,greference,gsimple_rev_cigar,gres,gsc, gdir);


    double end_gpu = get_time();


    printf("SW Time GPU: %.10lf\n", end_gpu - start_gpu);

    CHECK(cudaMemcpy(RES, gres, N * sizeof(int), cudaMemcpyDeviceToHost));

    CHECK(cudaMemcpy(SRC,gsimple_rev_cigar, N*2*S_LEN* sizeof(char), cudaMemcpyDeviceToHost));
    int t;
    for( t=0;t<N && res[t]==RES[t];t++);
        if(t==N)
        printf("VERIFICA!\n");

    for( int l=0;l<N;l++){
        for( t=0;SRC[512*2*l+t]!=0;t++){
            if(SRC[512*2*l+t]!=simple_rev_cigar[l][t])
                printf("LLL%d,%d,%d,%d\n",SRC[512*2*l+t],simple_rev_cigar[l][t],l,t);

        }


    }



    return 0;
}