within ;
model Bird
  constant Real mS = 3.0e-4;
  constant Real JS = 5.0e-9;
  constant Real mB = 4.5e-3;
  constant Real masstotal=mS+mB;
  constant Real JB = 7.0e-7;
  constant Real r0 = 2.5e-3;
  constant Real rS = 3.1e-3;
  constant Real hS = 5.8e-3;
  constant Real lS = 1.0e-2;
  constant Real lG = 1.5e-2;
  constant Real hB = 2.0e-2;
  constant Real lB = 2.01e-2;
  constant Real cp = 5.6e-3;
  constant Real g =  9.81;

  Real z(start=0);
  Real phi_b(start=-0.22169381);
  Real phi_s(start=-0.10344828);
  Real lamb_1(start=-0.16511245);
  Real lamb_2(start=-0.047088);

  Real lamb_1_sign(start=-1);
  parameter Integer state(start=2);

  Real zp, phi_bp, phi_sp, zpp, phi_bpp, phi_spp;

  Boolean transition_1(start=false);
  Boolean transition_2(start=false);
  Boolean transition_3(start=false);
  Boolean transition_4(start=false);
  Boolean transition_5(start=false);


  Real q[3, 1] = [z; phi_s; phi_b];
  Real qp[3, 1] = [zp; phi_sp; phi_bp];
  Real qpp[3, 1] = [zpp; phi_spp; phi_bpp];
  Real lamb[2, 1] = [lamb_1; lamb_2];

  // Real y[10, 1](start=[z; phi_s; phi_b; lamb_1; lamb_2; zp; phi_sp; phi_bp; lamb_1p; lamb_2p]);
  // Real yp[10, 1](start=[zp; phi_sp; phi_bp; lamb_1p; lamb_2p; zpp; phi_spp; phi_bpp; lamb_1pp; lamb_2pp]);

  constant Real M[3, 3] = [mS + mB, mB * lS, mB * lG;
                           mB * lS, JS + mB * lS^2, mB * lS * lG;
                           mB * lG, mB * lS * lG, JB + mB * lG^2];
  Real ff[3, 1];
  Real GT[3, 2];
  Real gvec[2, 1];

equation
  der(q) = qp;
  der(qp) = qpp;

  if state == 1 then
    // state 1
    GT =  [0, 0;
           1, 0;
           0, 1];
    gvec =  [lamb_1;
             lamb_2];
  elseif state == 2 then
    // state 2
    GT =  [0, 1;
           hS, rS;
           0, 0];
    gvec =  [rS - r0 + hS * phi_s;
             phi_sp + rS * phi_bp];
  elseif state == 3 then
    // state 3
    GT =  [0, 1;
           -hS, rS;
           0, 0];
    gvec =  [rS - r0 - hS * phi_s;
             phi_sp + rS * phi_bp];
  else
  end if;

  ff =  [- (mS + mB) * g;
        cp * (phi_b - phi_s) - mB * lS * g;
        cp * (phi_s - phi_b) - mB * lG * g];
  M * qpp = ff - GT * lamb;
  gvec = [0; 0];

  transition_1 =  (state == 1 and phi_bp < 0 and abs(hS * phi_s + (rS - r0)) < 1.0e-8);
  transition_2 =  (state == 1 and phi_bp > 0 and abs(hS * phi_s - (rS - r0)) < 1.0e-8);
  transition_3 =  (state == 2 and sign(lamb_1) == - sign(lamb_1_sign));
  transition_4 =  (state == 3 and phi_bp < 0 and sign(lamb_1) == - sign(lamb_1_sign));
  transition_5 =  (state == 3 and phi_bp > 0 and abs(hB * phi_b - lS + lG - lB - r0) < 1.0e-8);


  when transition_1 then
    state =  2;
  elsewhen transition_2 then
    state =  3;
  elsewhen transition_3 then
    state =  1;
    reinit(lamb_1_sign, -pre(lamb_1_sign));
  elsewhen transition_4 then
    state =  1;
    reinit(lamb_1_sign, -pre(lamb_1_sign));
  elsewhen transition_5 then
    reinit(phi_bp, -pre(phi_bp));
    state =  3;
  end when;

end Bird;
