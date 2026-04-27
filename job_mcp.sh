module load Core/Portland/nvhpc/24.7
nvcc -o main main.cc CheckGamma.cu
nohup ./main >  output.out &
