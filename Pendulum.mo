within ;
model Pendulum

  parameter Real perturbation = 0;

  constant Real pi = 2 * Modelica.Math.asin(1.0);

  constant Real phi = 2 * pi - 0.3;

  Real x(start=cos(phi + perturbation));
  Real y(start=sin(phi));

  Real vx, vy;

  parameter Real k = 100;

equation

  der(x) = vx;
  der(y) = vy;

  der(vx) = - x * k * (sqrt(x^2 + y^2) - 1) / sqrt(x^2 + y^2);
  der(vy) = - y * k * (sqrt(x^2 + y^2) - 1) / sqrt(x^2 + y^2) - 1;

end Pendulum;
