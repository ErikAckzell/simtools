within ;
model Squeezer
  import Modelica.Math.Matrices;
  import Modelica.Blocks.Math;

  constant Real m1=0.04325,m2=0.00365,m3=0.02373,m4=0.00706,m5=0.07050,m6=0.00706,m7=0.05498;
  constant Real i1=2.194e-6, i2=4.410e-7, i3=5.255e-6, i4=5.667e-7, i5=1.169e-5, i6=5.667e-7, i7=1.912e-5;
  constant Real xa=-0.06934,ya=-0.00227,xb=-0.03635, yb=0.03273,xc=0.014,yc=0.072;
  constant Real d=28.e-3, da=115.e-4, e=2.e-2, ea=1421.e-5, rr=7.e-3, ra=92.e-5;
  constant Real ss=35.e-3, sa=1874.e-5, sb=1043.e-5, sc=18.e-3, sd=2.e-2;
  constant Real ta=2308.e-5, tb=916.e-5, u=4.e-2, ua=1228.e-5, ub=449.e-5;
  constant Real zf=2.e-2, zt=4.e-2, fa=1421.e-5, mom=0.033, c0=4530., lo=0.07785;

  Real lambda[6, 1](start=[98.5668703962410896057654982170; -6.12268834425566265503114393122; 0; 0; 0; 0]);
  Real beta(start=-0.0617138900142764496358948458001);
  Real theta(start=0.0);
  Real gamma(start=0.455279819163070380255912382449);
  Real phi(start=0.222668390165885884674473185609);
  Real delta(start=0.487364979543842550225598953530);
  Real omega(start=-0.222668390165885884674473185609);
  Real epsilon(start=1.23054744454982119249735015568);
  Real q[7, 1] = [beta;theta;gamma;phi;delta;omega;epsilon];
  Real bep(start=0), thp(start=0), gap(start=0), php(start=0), dep(start=0), omp(start=0), epp(start=0);
  Real v[7, 1] = [bep; thp; gap; php; dep; omp; epp];

  Real w[7, 1];


  Real sibe=Modelica.Math.sin(beta);
  Real sith=Modelica.Math.sin(theta);
  Real siga=Modelica.Math.sin(gamma);
  Real siph=Modelica.Math.sin(phi);
  Real side=Modelica.Math.sin(delta);
  Real siom=Modelica.Math.sin(omega);
  Real siep=Modelica.Math.sin(epsilon);

  Real cobe=Modelica.Math.cos(beta);
  Real coth=Modelica.Math.cos(theta);
  Real coga=Modelica.Math.cos(gamma);
  Real coph=Modelica.Math.cos(phi);
  Real code=Modelica.Math.cos(delta);
  Real coom=Modelica.Math.cos(omega);
  Real coep=Modelica.Math.cos(epsilon);

  Real sibeth = Modelica.Math.sin(beta+theta);
  Real cobeth = Modelica.Math.cos(beta+theta);
  Real siphde = Modelica.Math.sin(phi+delta);
  Real cophde = Modelica.Math.cos(phi+delta);
  Real siomep = Modelica.Math.sin(omega+epsilon);
  Real coomep = Modelica.Math.cos(omega+epsilon);

  Real m[7,7]=[m1*ra^2 + m2*(rr^2-2*da*rr*coth+da^2) + i1 + i2, m2*(da^2-da*rr*coth) + i2, 0, 0, 0, 0, 0;
                        m2*(da^2-da*rr*coth) + i2, m2*da^2 + i2, 0, 0, 0, 0, 0;
                        0, 0, m3*(sa^2+sb^2) + i3, 0, 0, 0, 0;
                        0, 0, 0, m4*(e-ea)^2 + i4, m4*((e-ea)^2+zt*(e-ea)*siph) + i4, 0, 0;
                        0, 0, 0, m4*((e-ea)^2+zt*(e-ea)*siph) + i4, m4*(zt^2+2*zt*(e-ea)*siph+(e-ea)^2) + m5*(ta^2+tb^2)+ i4 + i5, 0, 0;
                        0, 0, 0, 0, 0, m6*(zf-fa)^2 + i6, m6*((zf-fa)^2-u*(zf-fa)*siom) + i6;
                        0, 0, 0, 0, 0, m6*((zf-fa)^2-u*(zf-fa)*siom) + i6, m6*((zf-fa)^2-2*u*(zf-fa)*siom+u^2) + m7*(ua^2+ub^2)+ i6 + i7];

  Real xd=sd*coga + sc*siga + xb, yd=sd*siga - sc*coga + yb, lang=sqrt((xd-xc)^2 + (yd-yc)^2);
  Real force=- c0 * (lang - lo)/lang, fx=force * (xd-xc), fy=force * (yd-yc);
  Real ff[7, 1]=[mom - m2*da*rr*thp*(thp+2*bep)*sith;
         m2*da*rr*bep^2*sith;
                 fx*(sc*coga - sd*siga) + fy*(sd*coga + sc*siga);
                 m4*zt*(e-ea)*dep^2*coph;
         - m4*zt*(e-ea)*php*(php+2*dep)*coph;
         - m6*u*(zf-fa)*epp^2*coom;
                 m6*u*(zf-fa)*omp*(omp+2*epp)*coom];

  Real gp[6, 7]=[- rr*sibe + d*sibeth, d*sibeth, - ss*coga, 0, 0, 0, 0;
                 rr*cobe - d*cobeth, - d*cobeth, - ss*siga, 0, 0, 0, 0;
                 - rr*sibe + d*sibeth, d*sibeth, 0, - e*cophde, - e*cophde + zt*side, 0, 0;
                 rr*cobe - d*cobeth, - d*cobeth, 0, - e*siphde, - e*siphde - zt*code, 0, 0;
                 - rr*sibe + d*sibeth, d*sibeth, 0, 0, 0, zf*siomep, zf*siomep - u*coep;
                 rr*cobe - d*cobeth, - d*cobeth, 0, 0, 0, - zf*coomep, - zf*coomep - u*siep];

  Real g[6, 1]=[rr*cobe - d*cobeth - ss*siga - xb;
                rr*sibe - d*sibeth + ss*coga - yb;
                rr*cobe - d*cobeth - e*siphde - zt*code - xa;
                rr*sibe - d*sibeth + e*cophde - zt*side - ya;
                rr*cobe - d*cobeth - zf*coomep - u*siep - xa;
                rr*sibe - d*sibeth - zf*siomep + u*coep - ya];
equation
  der(q) = v;
  der(v) = w;
  m * w = ff - transpose(gp)*lambda;
  zeros(6, 1) = g;


end Squeezer;
