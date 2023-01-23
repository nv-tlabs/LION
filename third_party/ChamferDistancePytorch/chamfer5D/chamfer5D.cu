
#include <stdio.h>
#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>



__global__ void NmDistanceKernel(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i){
	const int batch=2048;
	__shared__ float buf[batch*5];
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int k2=0;k2<m;k2+=batch){
			int end_k=min(m,k2+batch)-k2;
			for (int j=threadIdx.x;j<end_k*5;j+=blockDim.x){
				buf[j]=xyz2[(i*m+k2)*5+j];
			}
			__syncthreads();
			for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
				float x1=xyz[(i*n+j)*5+0];
				float y1=xyz[(i*n+j)*5+1];
				float r1=xyz[(i*n+j)*5+2];
				float g1=xyz[(i*n+j)*5+3];
				float b1=xyz[(i*n+j)*5+4];
				int best_i=0;
				float best=0;
				int end_ka=end_k-(end_k&5);
				if (end_ka==batch){
					for (int k=0;k<batch;k+=4){
						{
							float x2=buf[k*5+0]-x1;
							float y2=buf[k*5+1]-y1;
							float r2=buf[k*5+2]-r1;
							float g2=buf[k*5+3]-g1;
							float b2=buf[k*5+4]-b1;
							float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*5+5]-x1;
							float y2=buf[k*5+6]-y1;
							float r2=buf[k*5+7]-r1;
							float g2=buf[k*5+8]-g1;
							float b2=buf[k*5+9]-b1;
							float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*5+10]-x1;
							float y2=buf[k*5+11]-y1;
							float r2=buf[k*5+12]-r1;
							float g2=buf[k*5+13]-g1;
							float b2=buf[k*5+14]-b1;
							float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*5+15]-x1;
							float y2=buf[k*5+16]-y1;
							float r2=buf[k*5+17]-r1;
							float g2=buf[k*5+18]-g1;
							float b2=buf[k*5+19]-b1;
							float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}else{
					for (int k=0;k<end_ka;k+=4){
						{
							float x2=buf[k*5+0]-x1;
							float y2=buf[k*5+1]-y1;
							float r2=buf[k*5+2]-r1;
							float g2=buf[k*5+3]-g1;
							float b2=buf[k*5+4]-b1;
							float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
							if (k==0 || d<best){
								best=d;
								best_i=k+k2;
							}
						}
						{
							float x2=buf[k*5+5]-x1;
							float y2=buf[k*5+6]-y1;
							float r2=buf[k*5+7]-r1;
							float g2=buf[k*5+8]-g1;
							float b2=buf[k*5+9]-b1;
							float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
							if (d<best){
								best=d;
								best_i=k+k2+1;
							}
						}
						{
							float x2=buf[k*5+10]-x1;
							float y2=buf[k*5+11]-y1;
							float r2=buf[k*5+12]-r1;
							float g2=buf[k*5+13]-g1;
							float b2=buf[k*5+14]-b1;
							float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
							if (d<best){
								best=d;
								best_i=k+k2+2;
							}
						}
						{
							float x2=buf[k*5+15]-x1;
							float y2=buf[k*5+16]-y1;
							float r2=buf[k*5+17]-r1;
							float g2=buf[k*5+18]-g1;
							float b2=buf[k*5+19]-b1;
							float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
							if (d<best){
								best=d;
								best_i=k+k2+3;
							}
						}
					}
				}
				for (int k=end_ka;k<end_k;k++){
					float x2=buf[k*5+0]-x1;
					float y2=buf[k*5+1]-y1;
					float r2=buf[k*5+2]-r1;
					float g2=buf[k*5+3]-g1;
					float b2=buf[k*5+4]-b1;
					float d=x2*x2+y2*y2+r2*r2+g2*g2+b2*b2;
					if (k==0 || d<best){
						best=d;
						best_i=k+k2;
					}
				}
				if (k2==0 || result[(i*n+j)]>best){
					result[(i*n+j)]=best;
					result_i[(i*n+j)]=best_i;
				}
			}
			__syncthreads();
		}
	}
}
// int chamfer_cuda_forward(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i, cudaStream_t stream){
int chamfer_cuda_forward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor dist1, at::Tensor dist2, at::Tensor idx1, at::Tensor idx2){

	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A
	const auto m = xyz2.size(1); //num_points point cloud B

	NmDistanceKernel<<<dim3(32,16,1),512>>>(batch_size, n, xyz1.data<float>(), m, xyz2.data<float>(), dist1.data<float>(), idx1.data<int>());
	NmDistanceKernel<<<dim3(32,16,1),512>>>(batch_size, m, xyz2.data<float>(), n, xyz1.data<float>(), dist2.data<float>(), idx2.data<int>());

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd updateOutput: %s\n", cudaGetErrorString(err));
	    //THError("aborting");
	    return 0;
	  }
	  return 1;


}
__global__ void NmDistanceGradKernel(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,float * grad_xyz1,float * grad_xyz2){
	for (int i=blockIdx.x;i<b;i+=gridDim.x){
		for (int j=threadIdx.x+blockIdx.y*blockDim.x;j<n;j+=blockDim.x*gridDim.y){
			float x1=xyz1[(i*n+j)*5+0];
			float y1=xyz1[(i*n+j)*5+1];
			float r1=xyz1[(i*n+j)*5+2];
			float g1=xyz1[(i*n+j)*5+3];
			float b1=xyz1[(i*n+j)*5+4];
			int j2=idx1[i*n+j];
			float x2=xyz2[(i*m+j2)*5+0];
			float y2=xyz2[(i*m+j2)*5+1];
			float r2=xyz2[(i*m+j2)*5+2];
			float g2=xyz2[(i*m+j2)*5+3];
			float b2=xyz2[(i*m+j2)*5+4];
			float g=grad_dist1[i*n+j]*2;
			atomicAdd(&(grad_xyz1[(i*n+j)*5+0]),g*(x1-x2));
			atomicAdd(&(grad_xyz1[(i*n+j)*5+1]),g*(y1-y2));
			atomicAdd(&(grad_xyz1[(i*n+j)*5+2]),g*(r1-r2));
			atomicAdd(&(grad_xyz1[(i*n+j)*5+3]),g*(g1-g2));
			atomicAdd(&(grad_xyz1[(i*n+j)*5+4]),g*(b1-b2));
			atomicAdd(&(grad_xyz2[(i*m+j2)*5+0]),-(g*(x1-x2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*5+1]),-(g*(y1-y2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*5+2]),-(g*(r1-r2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*5+3]),-(g*(g1-g2)));
			atomicAdd(&(grad_xyz2[(i*m+j2)*5+4]),-(g*(b1-b2)));
		}
	}
}
// int chamfer_cuda_backward(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream){
int chamfer_cuda_backward(at::Tensor xyz1, at::Tensor xyz2, at::Tensor gradxyz1, at::Tensor gradxyz2, at::Tensor graddist1, at::Tensor graddist2, at::Tensor idx1, at::Tensor idx2){
	// cudaMemset(grad_xyz1,0,b*n*3*4);
	// cudaMemset(grad_xyz2,0,b*m*3*4);

	const auto batch_size = xyz1.size(0);
	const auto n = xyz1.size(1); //num_points point cloud A
	const auto m = xyz2.size(1); //num_points point cloud B

	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(batch_size,n,xyz1.data<float>(),m,xyz2.data<float>(),graddist1.data<float>(),idx1.data<int>(),gradxyz1.data<float>(),gradxyz2.data<float>());
	NmDistanceGradKernel<<<dim3(1,16,1),256>>>(batch_size,m,xyz2.data<float>(),n,xyz1.data<float>(),graddist2.data<float>(),idx2.data<int>(),gradxyz2.data<float>(),gradxyz1.data<float>());

	cudaError_t err = cudaGetLastError();
	  if (err != cudaSuccess) {
	    printf("error in nnd get grad: %s\n", cudaGetErrorString(err));
	    //THError("aborting");
	    return 0;
	  }
	  return 1;

}
