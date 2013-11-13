#include <algorithm>//All code done by Oleg K
#include <iostream>
#include <fstream>
#include <sstream>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <cmath>
#include <map>
#include <ctime>
#include <cuda.h>
#include <math_functions.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>
using namespace std;

#define _DTH cudaMemcpyDeviceToHost
#define _DTD cudaMemcpyDeviceToDevice
#define _HTD cudaMemcpyHostToDevice
#define INF (1<<28)
#define THREADS 256
#define SMALL_THREADS 64
#define DO_GPU 1

#define NUM_BOXES 50
#define NUM_COLORS 20

//Unthinking respect for authority is the greatest enemy of truth

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);

void _gen_random(int *AM, const int num_boxes,const int num_colors);

int cpu_version(const int *AM,const int num_boxes, const int num_colors,const int p_space){
	int ret=INF;
	int *Cost=(int *)malloc(num_boxes*num_colors*sizeof(int));
	int *DP=(int *)malloc(p_space*sizeof(int));
	memset(Cost,0,num_boxes*num_colors*sizeof(int));
	for(int i=0;i<num_boxes;i++)for(int j=0;j<num_colors;j++)for(int ii=0;ii<num_boxes;ii++)if(i!=ii){
		Cost[i*num_colors+j]+=AM[ii*num_colors+j];
	}
	for(int i=0;i<=num_boxes;i++)for(int j=0;j<(1<<num_colors);j++){
		DP[i*(1<<num_colors)+j]= (i==num_boxes && j==((1<<num_colors)-1)) ? 0:INF;
	}
	int idx;
	for(int i=num_boxes-1;i>=0;i--){
		for(int j=0;j<(1<<num_colors);j++){
			idx=i*(1<<num_colors)+j;
			DP[idx]=min(DP[idx],DP[(i+1)*(1<<num_colors)+j]);
			for(int k=0;k<num_colors;k++)if(!(j&(1<<k))){
				DP[idx]=min(DP[idx],Cost[i*num_colors+k]+DP[((i+1)*(1<<num_colors))+(j|(1<<k))]);
			}
		}
	}
	ret=DP[0];
	free(Cost);
	free(DP);
	return ret;
}

__global__ void mem_setup(int *DP, const int num_boxes, const int num_colors){//This can be sped up, PM me for details
	const int i=blockIdx.y;
	const int j=threadIdx.x+blockIdx.x*blockDim.x;
	const int m_bound=(1<<num_colors);
	if(j<m_bound){
		DP[i*m_bound+j]= (i==num_boxes && j==(m_bound-1)) ? 0:INF;
	}
}
__global__ void Cost_setup(const int *AM, int *Cost,const int num_boxes, const int num_colors){
	const int i=threadIdx.x+blockIdx.x*blockDim.x;
	const int j=blockIdx.y;
	const int k=blockIdx.z;
	if(i<num_boxes && i!=k){
		atomicAdd(&Cost[i*num_colors+j],AM[k*num_colors+j]);
	}
}
__global__ void GPU_version(const int *Cost, int *DP, const int ii, const int num_boxes, const int num_colors){
	const int j=threadIdx.x+blockIdx.x*blockDim.x;
	const int k=blockIdx.y;
	const int m_bound=(1<<num_colors);

	__shared__ int cur_cost;
	if(threadIdx.x==0){//Singapore person, I see you and think since you look at my code so often you should follow or star my work
		cur_cost=Cost[ii*num_colors+k];
	}
	__syncthreads();

	if(j<m_bound){
		const int idx=ii*m_bound+j;
		atomicMin(&DP[idx],DP[(ii+1)*m_bound+j]);
		if(!(j&(1<<k))){
			atomicMin(&DP[idx],cur_cost+DP[(ii+1)*m_bound+(j|(1<<k))]);
		}
	}
}

int main(){
	char ch;
	srand(time(NULL));
	const int num_boxes=NUM_BOXES;
	const int num_colors=NUM_COLORS;
	
	const int problem_space=(num_boxes+1)*(1<<num_colors);
	
	int *AM=(int *)malloc(num_boxes*num_colors*sizeof(int));

	_gen_random(AM,num_boxes,num_colors);

	int CPU_ans=INF,GPU_ans=INF;

	cout<<"\nRunning CPU implementation..\n";
    UINT wTimerRes = 0;
	DWORD CPU_time=0,GPU_time=0;
    bool init = InitMMTimer(wTimerRes);
    DWORD startTime=timeGetTime();

	CPU_ans=cpu_version(AM,num_boxes,num_colors,problem_space);

	DWORD endTime = timeGetTime();
    CPU_time=endTime-startTime;
    cout<<"CPU solution timing: "<<CPU_time<< " , answer= "<<CPU_ans<<'\n';
    DestroyMMTimer(wTimerRes, init);

	int compute_capability=0;
    cudaDeviceProp deviceProp;
    cudaError_t err=cudaGetDeviceProperties(&deviceProp, compute_capability);
    if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
    string ss= (deviceProp.major>=3 && deviceProp.minor>=5) ? "Capable!\n":"Not Sufficient compute capability!\n";
    cout<<ss;

	if(DO_GPU && (deviceProp.major>=3 && deviceProp.minor>=5)){// !(AMD || PYTHON || RUBY)
		const int num_bytes=problem_space*sizeof(int);
		const int num_bytesAM=num_boxes*num_colors*sizeof(int);
		const int m_bound=(1<<num_colors);
		int *D_AM,*D_Cost,*D_DP;
		err=cudaMalloc((void**)&D_AM,num_bytesAM);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&D_Cost,num_bytesAM);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&D_DP,num_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		dim3 memGrid((m_bound+THREADS-1)/THREADS,(num_boxes+1),1);
		dim3 costGrid((num_boxes+SMALL_THREADS-1)/SMALL_THREADS,num_colors,num_boxes);
		dim3 dpGrid((m_bound+THREADS-1)/THREADS,num_colors,1);
		int ii=num_boxes-1;
		//there is one thing we do know: that man is here for the sake of other men
		wTimerRes = 0;
        init = InitMMTimer(wTimerRes);
        startTime = timeGetTime();

		err=cudaMemset(D_Cost,0,num_bytesAM);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		
		err=cudaMemcpy(D_AM,AM,num_bytesAM,_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		mem_setup<<<memGrid,THREADS>>>(D_DP,num_boxes,num_colors);
		err = cudaThreadSynchronize();
        if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		Cost_setup<<<costGrid,SMALL_THREADS>>>(D_AM,D_Cost,num_boxes,num_colors);
		err = cudaThreadSynchronize();
        if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		for(;ii>=0;ii--){
			GPU_version<<<dpGrid,THREADS>>>(D_Cost,D_DP,ii,num_boxes,num_colors);
			err = cudaThreadSynchronize();
			if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		}
		err=cudaMemcpy(&GPU_ans,D_DP,sizeof(int),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		endTime = timeGetTime();
        GPU_time=endTime-startTime;
        cout<<"CUDA timing: "<<GPU_time<<" , answer= "<<GPU_ans<<'\n';
        DestroyMMTimer(wTimerRes, init);

		err=cudaFree(D_AM);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(D_Cost);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(D_DP);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	}

	free(AM);

	cin>>ch;
	return 0;
}

bool InitMMTimer(UINT wTimerRes){
	TIMECAPS tc;
	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) {return false;}
	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes); 
	return true;
}

void DestroyMMTimer(UINT wTimerRes, bool init){
	if(init)
		timeEndPeriod(wTimerRes);
}

void _gen_random(int *AM, const int num_boxes,const int num_colors){
	for(int i=0;i<num_boxes;i++){
		for(int j=0;j<num_colors;j++){
			AM[i*num_colors+j]= (rand()%4==0) ? 0:((rand()%127)+1);
		}
	}
}









