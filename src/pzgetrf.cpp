#include <ddla.h>
#include <cassert>
#include <vector>
#include <ddla_connector.h>
#include <ddla_utils.h>
#include <ddla_stream.h>
namespace DDLA{

void pzgetrf(
    const int& m, const int& n,
    std::complex<double>* d_A, const DDLA::DdlaDesc& array_descA,
    int* ipiv, // host
    int& info  // host
)
{
    DdlaHandle_t ddla_handle = array_descA.ddla_handle();

    MPI_Comm row_comm = ddla_handle->row_comm;
    MPI_Comm col_comm = ddla_handle->col_comm;

    ncclComm_t row_nccl_comm = ddla_handle->nccl_row_comm;
    ncclComm_t col_nccl_comm = ddla_handle->nccl_col_comm;

    int nprows = array_descA.nprows();
    int npcols = array_descA.npcols();
    int myprow = array_descA.myprow();
    int mypcol = array_descA.mypcol();

    int nb = array_descA.mb();
    int nb_real;
    assert(array_descA.mb()==array_descA.nb());
    int lld = array_descA.lld();

    int m_loc = array_descA.m_loc();
    int n_loc = array_descA.n_loc();

    deviceStream_t stream=ddla_handle->stream;
    deblasHandle_t blasH=ddla_handle->blasH;
    desolverHandle_t solverH=ddla_handle->solverH;
    

    int mm_row_start = 0;
    int mm_col_start = 0;
    int i_loc,j_loc;
    int owner_row,owner_col;

    int max_row;
    int max_prow;
    std::complex<double> max_value;

    std::complex<double> *d_max;
    DEVICE_CHECK(deviceMallocAsync(&d_max, sizeof(std::complex<double>)*nprows, stream));

    std::vector<int> h_id_max(nprows,0); // host

    std::complex<double> *d_temp;
    DEVICE_CHECK(deviceMallocAsync(&d_temp, sizeof(std::complex<double>)*n_loc, stream));

    std::complex<double> *d_temp_block;
    DEVICE_CHECK(deviceMallocAsync(&d_temp_block, sizeof(std::complex<double>)*nb*nb, stream));

    std::complex<double> *d_temp_L;
    DEVICE_CHECK(deviceMallocAsync(&d_temp_L, sizeof(std::complex<double>)*m_loc*nb, stream));

    std::complex<double> *d_temp_U;
    DEVICE_CHECK(deviceMallocAsync(&d_temp_U, sizeof(std::complex<double>)*nb*n_loc, stream));

    DEVICE_CHECK(deviceStreamSynchronize(stream));
    
    MPI_Barrier(MPI_COMM_WORLD);

    const std::complex<double> minus_one = {-1.0,0.0};
    const std::complex<double> one = {1.0,0.0};
    int one_int = 1;

    double time_for_pgetf2 = 0.0;
    double time_for_other = 0.0;
    // double time_for_max = 0.0;
    // double time_for_swap = 0.0;
    // double time_for_scal = 0.0;
    // double time_for_geru = 0.0;
    // double time_for_local_max = 0.0;
    // double time_for_global_max = 0.0;
    // double time_for_allreduce_device = 0.0;
    // double time_for_allreduce_host = 0.0;

    double start_time;

    for(int n_s=0;n_s<std::min(m,n);n_s+=nb){
        nb_real = std::min(nb, std::min(m,n)-n_s);

        i_loc = array_descA.indx_g2l_r(n_s);
        j_loc = array_descA.indx_g2l_c(n_s);

        owner_row = DDLA::indxg2p(n_s, nb, array_descA.irsrc(), nprows);
        owner_col = DDLA::indxg2p(n_s, nb, array_descA.icsrc(), npcols);

        
        // start pgetf2

        start_time = MPI_Wtime();
        // for(int i_tf2 = 0; i_tf2 < nb_real; i_tf2++){
        //     double start_time_tf2 = MPI_Wtime();
        //     DEVICE_CHECK(deviceMemsetAsync(d_max,0,nprows*sizeof(std::complex<double>),stream));
        //     // memset(h_max.data(),0,nprows*sizeof(std::complex<double>));
        //     memset(h_id_max.data(),0,nprows*sizeof(int));
        //     // printf("myid:%d, n_s:%d, i_tf2:%d\n",mpi_comm_global_h.myid,n_s,i_tf2);
        //     // find max_rows and value
        //     int i_panel, j_panel;
        //     if(i_loc >= 0)
        //         i_panel = i_loc + i_tf2;
        //     else
        //         i_panel = mm_row_start;
        //     if(j_loc >= 0)
        //         j_panel = j_loc + i_tf2;
        //     else
        //         j_panel = mm_col_start;
        //     if(j_loc >= 0){
        //         double start_time_local_max = MPI_Wtime();
        //         if(i_panel<m_loc){
        //             BLAS_CHECK(deblasIzamax(
        //                 blasH, m_loc-i_panel,
        //                 d_A + j_panel * lld + i_panel,1,
        //                 h_id_max.data()+myprow
        //             ));
        //             DEVICE_CHECK(deviceMemcpyAsync(
        //                 d_max+myprow, d_A + (i_panel + (h_id_max[myprow]-1) + j_panel * lld), sizeof(std::complex<double>),
        //                 deviceMemcpyDeviceToDevice, stream
        //             ));
        //         }
        //         // DEVICE_CHECK(cudaStreamSynchronize(stream));
        //         DEVICE_CHECK(deviceStreamSynchronize(stream));
        //         // MPI_Barrier(col_comm);
        //         // time_for_local_max += MPI_Wtime() - start_time_local_max;
        //         // double start_time_allreduce = MPI_Wtime();
        //         CCL_CHECK(ncclAllReduce(
        //             d_max, d_max, nprows * 2, ncclFloat64, ncclSum, col_nccl_comm, stream
        //         ));
        //         // device_stream.cudaSync();
        //         // MPI_Allreduce(MPI_IN_PLACE,h_max.data(),nprows,MPI_DOUBLE_COMPLEX,MPI_SUM,col_comm);
        //         // time_for_allreduce_device += MPI_Wtime() - start_time_allreduce;
        //         // double start_time_global_max = MPI_Wtime();
        //         BLAS_CHECK(deblasIzamax(blasH, nprows, d_max, 1, &max_prow));
        //         // max_prow = izamax_(&nprows, h_max.data(), &one_int);
        //         // device_stream.cudaSync();
        //         DEVICE_CHECK(deviceStreamSynchronize(stream));
        //         // time_for_global_max += MPI_Wtime() - start_time_global_max;
        //         // double start_time_allreduce_host = MPI_Wtime();
        //         max_prow--;
        //         h_id_max[myprow]=array_descA.indx_l2g_r(i_panel+h_id_max[myprow]-1);
        //         MPI_Allreduce(MPI_IN_PLACE,h_id_max.data(),nprows,MPI_INT,MPI_SUM,col_comm);
        //         // time_for_allreduce_host += MPI_Wtime() - start_time_allreduce_host;
        //         max_row = h_id_max[max_prow];
                
        //         DEVICE_CHECK(deviceMemcpyAsync(
        //             &max_value, d_max+max_prow, sizeof(std::complex<double>),
        //             deviceMemcpyDeviceToHost, stream
        //         ));
        //     }
        //     // DEVICE_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        //     // time_for_max += MPI_Wtime() - start_time_tf2;
            
        //     MPI_Bcast(&max_row, 1, MPI_INT, owner_col, row_comm);
        //     int max_loc_row = array_descA.indx_g2l_r(max_row);
        //     if(myprow == owner_row){
        //         ipiv[i_panel] = max_row + 1; // 1-based index like fortran
        //     }
        //     MPI_Bcast(&max_prow, 1, MPI_INT, owner_col, row_comm);
        //     start_time_tf2 = MPI_Wtime();
        //     // exchange rows
        //     if(owner_row == max_prow){
        //         if(myprow == owner_row && max_loc_row != i_panel)
        //             BLAS_CHECK(deblasZswap(
        //                 blasH, n_loc,
        //                 d_A + i_panel, lld,
        //                 d_A + max_loc_row, lld
        //             ));
        //     }else{
        //         if(myprow == owner_row){
        //             DEVICE_CHECK(deviceMemcpy2DAsync(
        //                 d_temp, sizeof(std::complex<double>),
        //                 d_A + i_panel, lld * sizeof(std::complex<double>),
        //                 sizeof(std::complex<double>), n_loc,
        //                 deviceMemcpyDeviceToDevice, stream
        //             ));
        //             // DEVICE_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        //             CCL_CHECK(
        //                 ncclSend(d_temp, n_loc * 2, ncclFloat64, max_prow, col_nccl_comm, stream)
        //             );
        //             CCL_CHECK(
        //                 ncclRecv(d_temp, n_loc  * 2, ncclFloat64, max_prow, col_nccl_comm, stream)
        //             );
        //             BLAS_CHECK(deblasZswap(
        //                 blasH, n_loc,
        //                 d_A + i_panel, lld,
        //                 d_temp, 1
        //             ));
        //             // DEVICE_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        //         }else if(myprow == max_prow){
        //             CCL_CHECK(
        //                 ncclRecv(d_temp, n_loc* 2, ncclFloat64, owner_row, col_nccl_comm, stream)
        //             );
        //             BLAS_CHECK(deblasZswap(
        //                 blasH, n_loc,
        //                 d_A + max_loc_row, lld,
        //                 d_temp, 1
        //             ));
        //             // DEVICE_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        //             CCL_CHECK(
        //                 ncclSend(d_temp, n_loc * 2, ncclFloat64, owner_row, col_nccl_comm, stream)
        //             );
        //             // DEVICE_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        //         }
        //     }
        //     // DEVICE_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        //     // time_for_swap += MPI_Wtime() - start_time_tf2;
        //     // start_time_tf2 = MPI_Wtime();
            
        //     // finish exchange rows
        //     MPI_Bcast(&max_value, sizeof(std::complex<double>), MPI_BYTE, owner_col, row_comm);
        //     if(std::abs(max_value)<1e-10){
        //         info = n_s+i_tf2+1;
        //         return;
        //     }
        //     // start reduce columns
        //     if(j_loc>=0){
        //         max_value = 1.0 / max_value; // inverse
        //         int64_t a_off;
        //         int length_row;
                
        //         if(i_loc>=0){
        //             a_off = (i_panel + 1) + j_panel * lld;
        //             length_row = m_loc - (i_panel + 1);
        //         }else{
        //             a_off = mm_row_start + j_panel * lld;
        //             length_row = m_loc - mm_row_start;
        //         }
        //         if(length_row>0){
        //             BLAS_CHECK(deblasZscal(
        //                 blasH, length_row,
        //                 &max_value,
        //                 d_A + a_off, 1
        //             ));
        //         }
        //         // DEVICE_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        //         // time_for_scal += MPI_Wtime() - start_time_tf2;
        //         // start_time_tf2 = MPI_Wtime();
        //         // printf("myid:%d, n_s:%d, i_tf2:%d, length_row:%d\n",mpi_comm_global_h.myid,n_s,i_tf2,length_row);
        //         int length_col = nb_real - i_tf2 - 1;
        //         if(myprow == owner_row){
        //             DEVICE_CHECK(deviceMemcpy2DAsync(
        //                 d_temp, 1 * sizeof(std::complex<double>),
        //                 d_A + i_panel + (j_panel + 1) * lld, lld * sizeof(std::complex<double>),
        //                 1*sizeof(std::complex<double>), length_col,
        //                 deviceMemcpyDeviceToDevice, stream
        //             ));
        //         }
        //         if(length_col>0)
        //             ncclBroadcast(d_temp,d_temp,length_col* 2,ncclFloat64,owner_row,col_nccl_comm,stream);
        //         // finish reduce columns
        //         // start update trailing matrix
                
        //         if(length_row>0&&length_row>0){
        //             BLAS_CHECK(deblasZgeru(
        //                 blasH, length_row, length_col,
        //                 &minus_one,
        //                 d_A + a_off, 1,
        //                 d_temp, 1,
        //                 d_A + a_off + lld, lld
        //             ));
        //         }
        //         // DEVICE_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        //         // time_for_geru += MPI_Wtime() - start_time_tf2;
        //     }
        //     // DEVICE_CHECK(cudaStreamSynchronize((cudaStream_t)stream));
        //     DEVICE_CHECK(deviceStreamSynchronize(stream));
        // }
        pzgetf2(
            m, nb_real,
            d_A, n_s, array_descA,
            ipiv, info
        );
        // finish pgetf2
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        time_for_pgetf2 += MPI_Wtime() - start_time;
        start_time = MPI_Wtime();
        // update trailing matrix
        if(i_loc>=0)
            mm_row_start +=nb; // update row start   
        if(j_loc>=0){
            mm_col_start+=nb;
            if(i_loc>=0){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp_block, nb_real * sizeof(std::complex<double>),
                    d_A + i_loc + j_loc * lld, lld * sizeof(std::complex<double>),
                    nb_real * sizeof(std::complex<double>), nb_real,
                    deviceMemcpyDeviceToDevice, stream
                ));
                
            }
            if(mm_row_start<m_loc){
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp_L, (m_loc-mm_row_start) * sizeof(std::complex<double>),
                    d_A + mm_row_start + j_loc * lld, lld * sizeof(std::complex<double>),
                    (m_loc - mm_row_start) * sizeof(std::complex<double>), nb_real,
                    deviceMemcpyDeviceToDevice, stream
                ));
            }
        }
        if(mm_row_start<m_loc){
            CCL_CHECK(ncclBroadcast(d_temp_L,d_temp_L,(m_loc - mm_row_start) * nb_real * 2,ncclFloat64,owner_col,row_nccl_comm,stream));
        }
        // broadcast block column
        if(i_loc>=0){
            CCL_CHECK(ncclBroadcast(d_temp_block,d_temp_block,nb_real * nb_real * 2,ncclFloat64,owner_col,row_nccl_comm,stream));
            if(mm_col_start<n_loc){
                BLAS_CHECK(deblasZtrsm(
                    blasH, DEBLAS_SIDE_LEFT, DEBLAS_FILL_MODE_LOWER, DEBLAS_OP_N, DEBLAS_DIAG_UNIT,
                    nb_real, n_loc - mm_col_start, &one,
                    d_temp_block, nb_real,
                    d_A + i_loc + mm_col_start * lld, lld)
                );
                DEVICE_CHECK(deviceMemcpy2DAsync(
                    d_temp_U, nb_real * sizeof(std::complex<double>),
                    d_A + i_loc + mm_col_start * lld, lld * sizeof(std::complex<double>),
                    nb_real * sizeof(std::complex<double>), n_loc - mm_col_start,
                    deviceMemcpyDeviceToDevice, stream
                ));
            }   
        }
        if(mm_col_start<n_loc){
            CCL_CHECK(ncclBroadcast(d_temp_U,d_temp_U,nb_real * (n_loc - mm_col_start) * 2,ncclFloat64,owner_row,col_nccl_comm,stream));
        }
        // printf("myid:%d, n_s:%d, update trailing matrix mm_row_start:%d, mm_col_start:%d\n",mpi_comm_global_h.myid,n_s,mm_row_start,mm_col_start);
        if(mm_row_start<m_loc&&mm_col_start<n_loc){
            BLAS_CHECK(deblasZgemm(
                blasH, DEBLAS_OP_N, DEBLAS_OP_N,
                m_loc - mm_row_start, n_loc - mm_col_start, nb_real,
                &minus_one,
                d_temp_L, m_loc - mm_row_start,
                d_temp_U, nb_real,
                &one,
                d_A + mm_row_start + mm_col_start * lld, lld
            ));
        }
        DEVICE_CHECK(deviceStreamSynchronize(stream));
        time_for_other += MPI_Wtime() - start_time;
    }
    info = 0;
    DEVICE_CHECK(deviceFreeAsync(d_max, stream));
    DEVICE_CHECK(deviceFreeAsync(d_temp, stream));
    DEVICE_CHECK(deviceFreeAsync(d_temp_block, stream));
    DEVICE_CHECK(deviceFreeAsync(d_temp_L, stream));
    DEVICE_CHECK(deviceFreeAsync(d_temp_U, stream));
    DEVICE_CHECK(deviceStreamSynchronize(stream));
    printf("myid:%d, pzgetrf time_for_pgetf2:%lf, time_for_other:%lf\n",ddla_handle->myid,time_for_pgetf2,time_for_other);

}

}