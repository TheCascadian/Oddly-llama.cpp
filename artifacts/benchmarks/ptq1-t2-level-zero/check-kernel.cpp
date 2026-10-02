#include "ptq1-t2.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>
int main() {
 try {
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
    std::mt19937 rng(1234);
    for (auto shape : {std::pair<int,int>{128,16}, {256,48}, {5120,128}, {256,17408}}) {
        const int K=shape.first,N=shape.second;
        std::vector<uint8_t> packed(N*(K/128)*28);
        std::vector<float> w(N*K);
        for(int n=0;n<N;++n) for(int g=0;g<K/128;++g) {
            auto *p=packed.data()+(n*(K/128)+g)*28;
            sycl::half h=sycl::half(float(1+rng()%10)/16); uint16_t hb=sycl::bit_cast<uint16_t>(h);
            p[26]=hb&255;p[27]=hb>>8;
            for(int b=0;b<26;++b) {
                int count=b<24?5:4, stride=b<16?16:b<24?8:2, first=b<16?b:b<24?80+b-16:120+b-24;
                unsigned v=0;
                for(int j=0;j<count;++j) {unsigned d=rng()%3;v=v*3+d;w[n*K+g*128+first+j*stride]=(int(d)-1)*float(h);}
                if(count==4)v*=3;p[b]=(v*256+242)/243;
            }
        }
        auto *dw=sycl::malloc_device<uint8_t>(ggml_sycl_t2_bytes(K,N),q);
        q.memcpy(dw,packed.data(),packed.size()).wait_and_throw();
        if(!ggml_sycl_t2_repack(q,dw,K,N)) return 4;
        for(int M : {1,2,3,4,5,8,9,16,33,128}) {
            std::vector<float> x(M*K),out(M*N);
            for(auto &v:x)v=(int(rng()%2001)-1000)/1000.f;
            if(M==1)std::fill(x.begin(),x.end(),0.f);
            auto *dx=sycl::malloc_device<float>(x.size(),q),*dy=sycl::malloc_device<float>(out.size(),q);
            auto *scratch=sycl::malloc_device<uint8_t>(ggml_sycl_t2_scratch_bytes(M,K),q);
            q.memcpy(dx,x.data(),x.size()*4).wait_and_throw();
            ggml_sycl_t2_mul_mat(q,dw,dx,K,dy,M,N,K,scratch);
            q.memcpy(out.data(),dy,out.size()*4).wait_and_throw();
            double se=0,ref2=0;
            for(int m=0;m<M;++m)for(int n=0;n<N;++n) {
                double ref=0;for(int k=0;k<K;++k)ref+=double(x[m*K+k])*w[n*K+k];
                double d=out[m*N+n]-ref;se+=d*d;ref2+=ref*ref;
            }
            double nmse=se/std::max(ref2,1e-30);
            std::cout<<"K="<<K<<" N="<<N<<" M="<<M<<" NMSE="<<nmse<<std::endl;
            if(!std::isfinite(nmse)||nmse>5e-4) return 5;
            sycl::free(dx,q);sycl::free(dy,q);sycl::free(scratch,q);
        }
        ggml_sycl_t2_restore(q,dw,K,N);
        std::vector<uint8_t> back(packed.size());q.memcpy(back.data(),dw,back.size()).wait_and_throw();
        if(back!=packed)return 6;
        packed[0]=1;q.memcpy(dw,packed.data(),packed.size()).wait_and_throw();
        if(ggml_sycl_t2_repack(q,dw,K,N))return 7;
        q.memcpy(back.data(),dw,back.size()).wait_and_throw();if(back!=packed)return 8;
        sycl::free(dw,q);
    }
    std::cout<<"PASS: numerical outputs, zero input, byte-exact restore, noncanonical fallback"<<std::endl;
 } catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 2;}
}
