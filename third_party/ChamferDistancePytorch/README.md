* adapted from https://github.com/ThibaultGROUEIX/ChamferDistancePytorch 

----------------------------------
# Pytorch Chamfer Distance.

Include a **CUDA** version, and a **PYTHON** version with pytorch standard operations.
NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt thresholds accordingly.

- [x] F - Score  



### CUDA VERSION

- [x] JIT compilation
- [x] Supports multi-gpu
- [x] 2D  point clouds.
- [x] 3D  point clouds.
- [x] 5D  point clouds.
- [x] Contiguous() safe.



### Python Version

- [x]  Supports any dimension



### Usage

```python
import torch, chamfer3D.dist_chamfer_3D, fscore
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
points1 = torch.rand(32, 1000, 3).cuda()
points2 = torch.rand(32, 2000, 3, requires_grad=True).cuda()
dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
f_score, precision, recall = fscore.fscore(dist1, dist2)
```



### Add it to your project as a submodule

```shell
git submodule add https://github.com/ThibaultGROUEIX/ChamferDistancePytorch
```



### Benchmark:  [forward + backward] pass
- [x] CUDA 10.1, NVIDIA 435, Pytorch 1.4
- [x] p1 : 32 x 2000 x dim
- [x] p2 : 32 x 1000 x dim

|  *Timing (sec * 1000)*  | 2D | 3D | 5D |
| ---------- | -------- | ------- | ------- |
| **Cuda Compiled**     | **1.2** | 1.4 |1.8 |
| **Cuda JIT**     | 1.3 | **1.4** |**1.5** |
| **Python**     | 37 | 37 | 37 |


| *Memory (MB)* |  2D | 3D | 5D |
| ---------- | -------- | ------- | ------- |
| **Cuda Compiled**     | 529 | 529  | 549 |
| **Cuda JIT**     | **520** | **529** |**549** |
| **Python**     | 2495 | 2495 | 2495 |



### What is the chamfer distance ? 

[Stanford course](http://graphics.stanford.edu/courses/cs468-17-spring/LectureSlides/L14%20-%203d%20deep%20learning%20on%20point%20cloud%20representation%20(analysis).pdf) on 3D deep Learning



### Aknowledgment 

Original backbone from [Fei Xia](https://github.com/fxia22/pointGAN/blob/master/nndistance/src/nnd_cuda.cu).

JIT cool trick from [Christian Diller](https://github.com/chrdiller)

### Troubleshoot

- `Undefined symbol: Zxxxxxxxxxxxxxxxxx `:

--> Fix: Make sure to `import torch` before you `import chamfer`.
--> Use pytorch.version >= 1.1.0

-  [RuntimeError: Ninja is required to load C++ extension](https://github.com/zhanghang1989/PyTorch-Encoding/issues/167)

```shell
wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
sudo unzip ninja-linux.zip -d /usr/local/bin/
sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force 
```





#### TODO:

* Discuss behaviour of torch.min() and tensor.min() which causes issues in some pytorch versions
