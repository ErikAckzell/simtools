	# Mechanical and geometric constants of the woodpecker model
	# according to 
	# Ch. Glocker: Dynamik von Starrkörpersystemen mit Reibung und Stössen
	# PhD thesis TU München, July 1995, pp. 162ff
	#
	mS = 3.0e-4 # Mass of sleeve [kg]
	JS = 5.0e-9 # Moment of inertia of the sleeve [kgm]
	mB = 4.5e-3 # Mass of bird [kg]
	masstotal=mS+mB # total mass
	JB = 7.0e-7 # Moment of inertia of bird [kgm]
	r0 = 2.5e-3 # Radius of the bar [m]
	rS = 3.1e-3 # Inner Radius of sleeve [m]
	hS = 5.8e-3 # 1/2 height of sleeve [m]
	lS = 1.0e-2 # verical distance sleeve origin to spring origin [m]
	lG = 1.5e-2 # vertical distance spring origin to bird origin [m]
	hB = 2.0e-2 # y coordinate beak (in bird coordinate system) [m]
	lB = 2.01e-2 # -x coordinate beak (in bird coordinate system) [m]
	cp = 5.6e-3 # rotational spring constant [N/rad]
	g  = 9.81 #  [m/s^2]
