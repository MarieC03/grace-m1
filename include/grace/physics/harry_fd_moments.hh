#ifndef GRACE_HARRY_FD_MOMENTS_HH
#define GRACE_HARRY_FD_MOMENTS_HH
#include <Kokkos_Core.hpp>
namespace grace {
// F_k(eta)=integral x^k/(1+exp(x-eta)) dx. For eta<=0 return
// exp(-eta)*F_k to avoid underflow. Coefficients interpolate
// -k! Li_(k+1)(-z)/z on z in [0,1]; positive eta uses exact
// integer-order reflection identities. See generate_fd_moments.py.
struct harry_fd_pair { double f2,f3; };
KOKKOS_INLINE_FUNCTION harry_fd_pair harry_fd_moments(double eta) {
    constexpr double c2[16]={1.8959267521834093,-0.098018361485851455,0.0055751856889071135,-0.0004347455103064026,4.0269101885183413e-05,-4.160151006424715e-06,4.6356641412077862e-07,-5.4613710313998696e-08,6.7153081819531755e-09,-8.5419873113412618e-10,1.1169311314481524e-10,-1.4942602303974298e-11,2.0379855936668267e-12,-2.8257383330025779e-13,3.9725853981842328e-14,-5.5413122910129397e-15};
    constexpr double c3[16]={5.8348077987790052,-0.15852120742430029,0.0062617733243248708,-0.00037776318766703267,2.8669638031124224e-05,-2.5151078701991834e-06,2.4389899191172403e-07,-2.5458761679622659e-08,2.8118625641606822e-09,-3.2478029227228059e-10,3.8903043316086891e-11,-4.8026329845084365e-12,6.0816637518896589e-13,-7.8705879010812181e-14,1.0375640732596532e-14,-1.3653574253107075e-15};
    double const z=Kokkos::exp(-Kokkos::abs(eta)),x=2*z-1;
    double b21=0,b22=0,b31=0,b32=0;
    for(int i=15;i>0;--i) {
        double a=2*x*b21-b22+c2[i];b22=b21;b21=a;
        a=2*x*b31-b32+c3[i];b32=b31;b31=a;
    }
    double const s2=x*b21-b22+c2[0],s3=x*b31-b32+c3[0];
    if(eta<=0)return {s2,s3};
    constexpr double pi2=9.8696044010893586188;
    double const e2=eta*eta;
    return {eta*(e2+pi2)/3+z*s2,e2*(e2/4+pi2/2)+7*pi2*pi2/60-z*s3};
}
} // namespace grace
#endif
