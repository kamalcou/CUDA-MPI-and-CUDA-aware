#!/bin/bash
#SBATCH --qos=class  
#SBATCH --job-name=HW4_Test
#SBATCH --output=HW4-%J.out
#SBATCH --nodes=1-1  
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=2000mb            # memory per node in MB 
#SBATCH --requeue
#SBATCH --time=2:00:00 
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=XYZ@crimson.ua.edu

source /opt/asn/etc/asn-bash-profiles-special/modules.sh
module load openmpi/4.1.4-gcc11-cuda
#module load openmpi/4.1.3-gcc9
# module load cuda/11.7.0
##export OMPI_MCA_btl=^openib
mpicc -Wall -O -o mpi_life_1d mpi_life_1d.c
mpirun -np 2 ./mpi_life_1d 10000 5000 /scratch/ualclsb0056
mpirun -np 2 ./mpi_life_1d 10000 5000 /scratch/ualclsb0056
mpirun -np 2 ./mpi_life_1d 10000 5000 /scratch/ualclsb0056
# mpirun -np 2 ./mpi_life_1d 10000 5000 /scratch/ualclsb0056
#nvcc -O3  -I/opt/asn/apps/openmpi_4.1.4_gcc11_cuda/include -DDEBUG0 -DDEBUG2 -o cuda_MPI_life cuda_MPI_life.cu -lmpi
nvcc -O3  -I/opt/asn/apps/openmpi_4.1.4_gcc11_cuda/include -o cuda_MPI_life cuda_MPI_life.cu -lmpi

mpirun -np 2  ./cuda_MPI_life 10000 5000 /scratch/ualclsb0056
mpirun -np 2  ./cuda_MPI_life 10000 5000 /scratch/ualclsb0056
mpirun -np 2  ./cuda_MPI_life 10000 5000 /scratch/ualclsb0056


nvcc -O3  -I/opt/asn/apps/openmpi_4.1.4_gcc11_cuda/include -o cuda_aware_MPI_life cuda_aware_MPI_life.cu -lmpi
#mpirun -np 2  ./mpi_life 5 2 output
mpirun -np 2  ./cuda_aware_MPI_life 10000 5000 /scratch/ualclsb0056
mpirun -np 2  ./cuda_aware_MPI_life 10000 5000 /scratch/ualclsb0056
mpirun -np 2  ./cuda_aware_MPI_life 10000 5000 /scratch/ualclsb0056
