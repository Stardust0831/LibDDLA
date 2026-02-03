#include <ddla_desc.h>
#include <ddla_stream.h>

namespace DDLA{

void DdlaDesc::init_square_blk(const int &m, const int &n, const int &irsrc, const int &icsrc)
{
    int mb = std::ceil(double(m) / nprows_);
    int nb = std::ceil(double(n) / npcols_);
    int nb_real = std::min(mb, nb); // use the smaller block size
    this->init(m, n, nb_real, nb_real, irsrc, icsrc);
    return;
}

DdlaDesc::DdlaDesc(const DdlaHandle_t& ddla_handle)
{
    this->ddla_handle_ = ddla_handle;
    this->nprows_ = ddla_handle->nprows_;
    this->npcols_ = ddla_handle->npcols_;
    this->myprow_ = ddla_handle->myprow_;
    this->mypcol_ = ddla_handle->mypcol_;
}

void DdlaDesc::init(const int &m, const int &n, const int &mb, const int &nb, const int &irsrc, const int &icsrc){
    this->m_ = m;
    this->n_ = n;
    this->mb_ = mb;
    this->nb_ = nb;
    this->irsrc_ = irsrc;
    this->icsrc_ = icsrc;
    // compute local sizes
    this->m_local_ = DDLA::num_loc(m, mb, myprow_, irsrc_, nprows_);
    this->n_local_ = DDLA::num_loc(n, nb, mypcol_, icsrc_, npcols_);
    this->lld_ = std::max(this->m_local_,1);
    return;
}

int DdlaDesc::indx_g2l_r(int gindx) const{
    if(this->myprow_ != DDLA::indxg2p(gindx, this->mb_, this->irsrc_, this->nprows_) || gindx >= this->m_)
        return -1;
    return DDLA::indxg2l(gindx, this->mb_, this->nprows_);
}


}