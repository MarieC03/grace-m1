
/****************************************************************************/
/*            Resistive (REGIME) first-order helpers, SymPy generated       */
/*  sqrtgF index map: 0:F^{0x} 1:F^{0y} 2:F^{0z} 3:F^{xy} 4:F^{xz} 5:F^{yz}  */
/*  dsqrtgF_dx[c + 6*j] = d_j sqrtgF[c]                                      */
/****************************************************************************/
#ifndef GRACE_RESISTIVE_SUBEXPR_HH
#define GRACE_RESISTIVE_SUBEXPR_HH

#include <Kokkos_Core.hpp>
static void KOKKOS_INLINE_FUNCTION
resistive_get_sqrtgF(
	double alp,
	const double Bd[4],
	const double wd[4],
	double (*sqrtgF)[6]
)
{
	double x0 = 1/(alp);
	(*sqrtgF)[0] = x0*(Bd[2]*wd[3] - Bd[3]*wd[2]);
	(*sqrtgF)[1] = -x0*(Bd[1]*wd[3] - Bd[3]*wd[1]);
	(*sqrtgF)[2] = x0*(Bd[1]*wd[2] - Bd[2]*wd[1]);
	(*sqrtgF)[3] = x0*(Bd[0]*wd[3] - Bd[3]*wd[0]);
	(*sqrtgF)[4] = -x0*(Bd[0]*wd[2] - Bd[2]*wd[0]);
	(*sqrtgF)[5] = x0*(Bd[0]*wd[1] - Bd[1]*wd[0]);
}

static void KOKKOS_INLINE_FUNCTION
resistive_get_divF(
	double sqrtg,
	const double dsqrtgF_dx[18],
	double (*divF)[4]
)
{
	double x0 = 1/(sqrtg);
	(*divF)[0] = x0*(dsqrtgF_dx[0] + dsqrtgF_dx[14] + dsqrtgF_dx[7]);
	(*divF)[1] = x0*(dsqrtgF_dx[16] + dsqrtgF_dx[9]);
	(*divF)[2] = x0*(dsqrtgF_dx[17] - dsqrtgF_dx[3]);
	(*divF)[3] = -x0*(dsqrtgF_dx[11] + dsqrtgF_dx[4]);
}

static void KOKKOS_INLINE_FUNCTION
resistive_get_E1(
	double alp,
	const double uu[4],
	const double ud[4],
	const double nd[4],
	const double divF[4],
	double (*E1u)[4]
)
{
	double x0 = divF[0]*ud[0] + divF[1]*ud[1] + divF[2]*ud[2] + divF[3]*ud[3];
	double x1 = divF[0] + uu[0]*x0;
	double x2 = divF[1] + uu[1]*x0;
	double x3 = divF[2] + uu[2]*x0;
	double x4 = divF[3] + uu[3]*x0;
	double x5 = nd[0]*x1 + nd[1]*x2 + nd[2]*x3 + nd[3]*x4;
	double x6 = alp*uu[0];
	(*E1u)[0] = uu[0]*(alp*x1 + x5);
	(*E1u)[1] = uu[1]*x5 + x2*x6;
	(*E1u)[2] = uu[2]*x5 + x3*x6;
	(*E1u)[3] = uu[3]*x5 + x4*x6;
}

#endif
